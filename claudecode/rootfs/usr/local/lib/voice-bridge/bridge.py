"""Claude Code voice bridge, inside the Claude Code add-on.

HA (Assist scripts) → this service → Claude Agent SDK sessions in /homeassistant.

  POST /tasks                    {project, prompt, source}   start a task
  POST /tasks/{ref}/message      {text}                      add to a task
  GET  /tasks                                                status of tasks
  POST /tasks/{ref}/permission   {allow, req_id?}            answer a prompt

{ref} is a task id ("t3"), a project name/alias, or "latest".

Events go back to HA's webhook as JSON {"event": ..., ...}:
  task_finished       a task went idle (result text, cost, duration)
  permission_request  a tool needs approval (actionable push in HA)

Scope: one project, the HA config. Nothing here reaches outside this add-on
except HA itself (webhook + the homeassistant MCP server).

Auth: the caller must be on the Supervisor's hassio network (172.30.32.0/23;
the port is not published to the host), and every request needs
"Authorization: Bearer <token>".

Config comes from the environment, set by the add-on's startup from its options:
  VOICE_BRIDGE_TOKEN       bearer token HA sends (required)
  VOICE_BRIDGE_WEBHOOK_ID  HA webhook id for events (required)
  VOICE_BRIDGE_PORT        default 8791
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolPermissionContext,
    ToolUseBlock,
)

TOKEN = os.environ["VOICE_BRIDGE_TOKEN"]
WEBHOOK_URL = f"http://homeassistant:8123/api/webhook/{os.environ['VOICE_BRIDGE_WEBHOOK_ID']}"
PORT = int(os.environ.get("VOICE_BRIDGE_PORT", 8791))
CLAUDE_PATH = shutil.which("claude") or "/usr/local/bin/claude"
# /data is the add-on's private storage: survives restarts, not in HA's config dir.
STATE_FILE = Path("/data/voice_bridge_tasks.json")
# HA Core reaches add-ons from the hassio bridge network.
ALLOWED_NET = ipaddress.ip_network("172.30.32.0/23")

PROJECTS = {
    "homeassistant": {"path": "/homeassistant", "aliases": ["home assistant", "ha", "home", "house"]},
}

# How long an idle task keeps its CLI process; later messages resume it from
# the saved session. Kept short: each idle CLI holds ~100-300 MB, and on a 4 GB
# Pi a couple of them got the interactive terminal session OOM-killed.
IDLE_DISCONNECT_S = 2 * 60
# Unanswered permission prompts are denied after this.
PERMISSION_TIMEOUT_S = 30 * 60
# After a result, wait this long for queued messages before calling it done.
SETTLE_S = 3

VOICE_PROMPT = """\
This task was started by voice through Home Assistant; the user is not at the
keyboard. You run inside the Claude Code add-on on the Home Assistant server and
work only on this house's Home Assistant setup. When you finish, your final
message is summarised by a small local model and read aloud, so end with a short
plain-language summary of what you did or found (no tables, no code) and anything
the user needs to decide. Before finishing a config change, validate it
(`ha core check`) and reload or say that a restart is needed.
Tools that need approval are sent to the user's phone; if one is denied, carry
on without it and say what you would have done. Don't use AskUserQuestion —
pick the sensible default and mention it, or stop and ask in your final message.
A final message that ends with a question mark is asked aloud and the user's
spoken answer comes back to you, so end with a question only when you need an
answer to continue, and never close with offers like "Want me to…?".
Spoken answers go through speech-to-text: read them charitably.
"""

log = logging.getLogger("voice-bridge")


@dataclass
class Task:
    id: str
    project: str
    cwd: str
    prompt: str
    source: str = "voice"
    status: str = "starting"  # starting | running | waiting_permission | idle | error | interrupted
    session_id: str | None = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    last_text: str = ""  # latest assistant text (progress)
    result: str = ""  # final result of the last turn
    current_tool: str = ""
    cost_usd: float = 0.0
    turns: int = 0
    messages: list[str] = field(default_factory=list)  # follow-ups sent
    pending_permission: dict | None = None

    def public(self) -> dict:
        d = asdict(self)
        d["age_min"] = round((time.time() - self.created) / 60)
        d["since_update_min"] = round((time.time() - self.updated) / 60)
        return d


class Runner:
    """Owns one SDK client per live task."""

    def __init__(self, task: Task, bridge: "Bridge") -> None:
        self.task = task
        self.bridge = bridge
        self.client: ClaudeSDKClient | None = None
        self.reader: asyncio.Task | None = None
        self.outstanding = 0  # queries sent but not yet answered
        self.idle_timer: asyncio.TimerHandle | None = None
        self.settle_task: asyncio.Task | None = None
        self.permission_future: asyncio.Future | None = None
        self.lock = asyncio.Lock()

    def _options(self) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            cwd=self.task.cwd,
            cli_path=CLAUDE_PATH,
            permission_mode="acceptEdits",
            can_use_tool=self._can_use_tool,
            # User + project settings: CLAUDE.md, MCP servers (HA), allow-lists.
            setting_sources=["user", "project", "local"],
            system_prompt={"type": "preset", "preset": "claude_code", "append": VOICE_PROMPT},
            resume=self.task.session_id,
            stderr=lambda line: log.debug("[%s] %s", self.task.id, line),
        )

    async def send(self, text: str) -> None:
        async with self.lock:
            if self.client is None:
                self.client = ClaudeSDKClient(self._options())
                await self.client.connect()
                self.reader = asyncio.create_task(self._read())
            self._cancel_idle()
            if self.settle_task:
                self.settle_task.cancel()
                self.settle_task = None
            self.outstanding += 1
            self.bridge.update(self.task, status="running")
            await self.client.query(text)

    async def _read(self) -> None:
        assert self.client
        try:
            async for msg in self.client.receive_messages():
                self._handle(msg)
        except Exception as e:  # CLI died, auth failure, etc.
            log.exception("[%s] reader failed", self.task.id)
            self.bridge.update(self.task, status="error", result=f"Bridge error: {e}")
            await self.bridge.notify_finished(self.task)
            await self.close()

    def _handle(self, msg: Any) -> None:
        t = self.task
        if isinstance(msg, SystemMessage) and msg.subtype == "init":
            sid = msg.data.get("session_id")
            if sid:
                self.bridge.update(t, session_id=sid)
        elif isinstance(msg, AssistantMessage):
            texts = [b.text for b in msg.content if isinstance(b, TextBlock) and b.text.strip()]
            tools = [b.name for b in msg.content if isinstance(b, ToolUseBlock)]
            changes: dict[str, Any] = {}
            if texts:
                changes["last_text"] = texts[-1][-2000:]
            if tools:
                changes["current_tool"] = tools[-1]
            if changes:
                self.bridge.update(t, **changes)
        elif isinstance(msg, ResultMessage):
            # No query outstanding → Claude carried on by itself (e.g. a
            # background command finished), not an answer to the user.
            unprompted = self.outstanding == 0
            self.outstanding = max(0, self.outstanding - 1)
            self.bridge.update(
                t,
                session_id=msg.session_id,
                result=(msg.result or t.last_text or "")[-4000:],
                cost_usd=round(t.cost_usd + (msg.total_cost_usd or 0), 4),
                turns=t.turns + (msg.num_turns or 0),
                current_tool="",
            )
            if msg.is_error:
                self.bridge.update(t, status="error")
            # A follow-up may have been merged into this turn or still be
            # queued; give the CLI a moment before announcing.
            if self.settle_task:
                self.settle_task.cancel()
            self.settle_task = asyncio.create_task(self._settle(msg.is_error, unprompted))

    async def _settle(self, is_error: bool, unprompted: bool) -> None:
        await asyncio.sleep(SETTLE_S)
        if self.outstanding > 0:
            # Merged turns answer several queries with one result.
            log.info("[%s] %d queued message(s) unanswered after result; treating as merged",
                     self.task.id, self.outstanding)
            self.outstanding = 0
        self.bridge.update(self.task, status="error" if is_error else "idle")
        await self.bridge.notify_finished(self.task, unprompted)
        self._arm_idle()

    async def _can_use_tool(
        self, tool: str, tool_input: dict[str, Any], ctx: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        if tool == "AskUserQuestion":
            return PermissionResultDeny(
                message="The user is away from the keyboard. Pick the recommended "
                "option and say so, or end your turn with the question."
            )
        req_id = f"{self.task.id}-{int(time.time() * 1000) % 10_000_000}"
        summary = describe_tool(tool, tool_input)
        self.permission_future = asyncio.get_running_loop().create_future()
        self.bridge.update(
            self.task,
            status="waiting_permission",
            pending_permission={"req_id": req_id, "tool": tool, "summary": summary},
        )
        await self.bridge.post_ha({
            "event": "permission_request",
            "task_id": self.task.id,
            "project": self.task.project,
            "source": self.task.source,
            "req_id": req_id,
            "tool": tool,
            "summary": summary,
            "title": short_title(self.task.prompt),
        })
        try:
            allow = await asyncio.wait_for(self.permission_future, PERMISSION_TIMEOUT_S)
        except asyncio.TimeoutError:
            allow = False
        finally:
            self.permission_future = None
            self.bridge.update(self.task, status="running", pending_permission=None)
        if allow:
            return PermissionResultAllow()
        return PermissionResultDeny(message="The user declined this (or didn't answer). Carry on without it.")

    def answer_permission(self, allow: bool) -> bool:
        if self.permission_future and not self.permission_future.done():
            self.permission_future.set_result(allow)
            return True
        return False

    def _arm_idle(self) -> None:
        self._cancel_idle()
        loop = asyncio.get_running_loop()
        self.idle_timer = loop.call_later(IDLE_DISCONNECT_S, lambda: asyncio.create_task(self.close()))

    def _cancel_idle(self) -> None:
        if self.idle_timer:
            self.idle_timer.cancel()
            self.idle_timer = None

    async def close(self) -> None:
        self._cancel_idle()
        client, self.client = self.client, None
        if self.reader and self.reader is not asyncio.current_task():
            self.reader.cancel()
        self.reader = None
        if client:
            try:
                await client.disconnect()
            except Exception:
                log.debug("[%s] disconnect failed", self.task.id, exc_info=True)


class Bridge:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.runners: dict[str, Runner] = {}
        self.next_id = 1
        self.http: aiohttp.ClientSession | None = None
        self._load()

    # --- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not STATE_FILE.exists():
            return
        data = json.loads(STATE_FILE.read_text())
        self.next_id = data.get("next_id", 1)
        for d in data.get("tasks", []):
            t = Task(**d)
            if t.status in ("starting", "running", "waiting_permission"):
                t.status = "interrupted"  # bridge restarted mid-task
                t.pending_permission = None
            self.tasks[t.id] = t

    def _save(self) -> None:
        # Keep the last 50 tasks.
        keep = sorted(self.tasks.values(), key=lambda t: t.created)[-50:]
        STATE_FILE.write_text(json.dumps(
            {"next_id": self.next_id, "tasks": [asdict(t) for t in keep]}, indent=1))

    def update(self, t: Task, **changes: Any) -> None:
        for k, v in changes.items():
            setattr(t, k, v)
        t.updated = time.time()
        self._save()

    # --- lookups ---------------------------------------------------------
    def resolve_project(self, name: str) -> tuple[str, str] | None:
        n = (name or "").strip().lower()
        if not n:  # one project; an empty name means it
            key = next(iter(PROJECTS))
            return key, PROJECTS[key]["path"]
        for key, p in PROJECTS.items():
            if n == key or n in p.get("aliases", []) or any(a in n for a in [key, *p.get("aliases", [])]):
                return key, p["path"]
        return None

    def find(self, ref: str | None) -> Task | None:
        ordered = sorted(self.tasks.values(), key=lambda t: t.updated, reverse=True)
        if not ordered:
            return None
        r = (ref or "latest").strip().lower()
        if r in ("", "latest", "last", "current"):
            live = [t for t in ordered if t.status in ("running", "waiting_permission", "starting")]
            return (live or ordered)[0]
        if r in self.tasks:
            return self.tasks[r]
        if r.isdigit() and f"t{r}" in self.tasks:
            return self.tasks[f"t{r}"]
        proj = self.resolve_project(r)
        if proj:
            for t in ordered:
                if t.project == proj[0]:
                    return t
        # Match words of the original request.
        for t in ordered:
            if r in t.prompt.lower():
                return t
        return None

    # --- actions ---------------------------------------------------------
    async def start(self, project: str, prompt: str, source: str) -> Task:
        proj = self.resolve_project(project)
        if not proj:
            raise web.HTTPBadRequest(text=json.dumps({
                "error": f"Unknown project '{project}'",
                "projects": list(PROJECTS),
            }), content_type="application/json")
        t = Task(id=f"t{self.next_id}", project=proj[0], cwd=proj[1], prompt=prompt, source=source)
        self.next_id += 1
        self.tasks[t.id] = t
        self._save()
        runner = self.runners[t.id] = Runner(t, self)
        await runner.send(prompt)
        return t

    async def message(self, t: Task, text: str) -> str:
        self.update(t, messages=[*t.messages, text][-20:])
        runner = self.runners.get(t.id)
        if runner is None:
            runner = self.runners[t.id] = Runner(t, self)
        was_live = t.status in ("running", "waiting_permission")
        await runner.send(text)
        return "queued into the running task" if was_live else "resumed the task"

    async def notify_finished(self, t: Task, unprompted: bool = False) -> None:
        await self.post_ha({
            "event": "task_finished",
            "task_id": t.id,
            "project": t.project,
            "source": t.source,
            "title": short_title(t.prompt),
            "prompt": t.prompt,
            "status": t.status,
            "unprompted": unprompted,
            "result": t.result,
            "cost_usd": t.cost_usd,
            "minutes": round((time.time() - t.created) / 60, 1),
            "session_id": t.session_id,
            "cwd": t.cwd,
        })

    async def post_ha(self, payload: dict) -> None:
        try:
            assert self.http
            async with self.http.post(WEBHOOK_URL, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status >= 300:
                    log.warning("HA webhook %s → %s", payload.get("event"), r.status)
        except Exception:
            log.exception("HA webhook post failed (%s)", payload.get("event"))


def short_title(prompt: str, words: int = 8) -> str:
    w = prompt.split()
    return " ".join(w[:words]) + ("…" if len(w) > words else "")


def describe_tool(tool: str, tool_input: dict[str, Any]) -> str:
    """One line a person can approve from a lock screen."""
    if tool == "Bash":
        return f"Run: {tool_input.get('command', '')[:300]}"
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        return f"{tool} {tool_input.get('file_path') or tool_input.get('notebook_path', '')}"
    if tool == "WebFetch":
        return f"Fetch {tool_input.get('url', '')}"
    if tool.startswith("mcp__"):
        parts = tool.split("__")
        name = parts[-1]
        args = json.dumps(tool_input, ensure_ascii=False)
        return f"{name} {args[:250]}"
    return f"{tool} {json.dumps(tool_input, ensure_ascii=False)[:250]}"


# --- HTTP ------------------------------------------------------------------

@web.middleware
async def auth(request: web.Request, handler):
    # Source network first (only the hassio network can route here), then token.
    try:
        allowed = ipaddress.ip_address(request.remote or "") in ALLOWED_NET
    except ValueError:
        allowed = False
    if not allowed:
        log.warning("rejected %s %s from %s", request.method, request.path, request.remote)
        raise web.HTTPForbidden()
    if request.headers.get("Authorization") != f"Bearer {TOKEN}":
        raise web.HTTPUnauthorized()
    return await handler(request)


def routes(bridge: Bridge) -> web.RouteTableDef:
    r = web.RouteTableDef()

    def need(t: Task | None, ref: str) -> Task:
        if t is None:
            raise web.HTTPNotFound(text=json.dumps({"error": f"No task matching '{ref}'"}),
                                   content_type="application/json")
        return t

    @r.get("/tasks")
    async def list_tasks(request: web.Request):
        ordered = sorted(bridge.tasks.values(), key=lambda t: t.updated, reverse=True)
        limit = int(request.query.get("limit", 5))
        return web.json_response({"tasks": [t.public() for t in ordered[:limit]],
                                  "projects": list(PROJECTS)})

    @r.post("/tasks")
    async def start(request: web.Request):
        body = await request.json()
        t = await bridge.start(body.get("project", ""), body["prompt"], body.get("source", "voice"))
        return web.json_response(t.public())

    @r.get("/tasks/{ref}")
    async def get_task(request: web.Request):
        ref = request.match_info["ref"]
        return web.json_response(need(bridge.find(ref), ref).public())

    @r.post("/tasks/{ref}/message")
    async def message(request: web.Request):
        ref = request.match_info["ref"]
        t = need(bridge.find(ref), ref)
        body = await request.json()
        how = await bridge.message(t, body["text"])
        return web.json_response({**t.public(), "delivery": how})

    @r.post("/tasks/{ref}/permission")
    async def permission(request: web.Request):
        ref = request.match_info["ref"]
        body = await request.json()
        t = need(bridge.find(ref), ref)
        pending = t.pending_permission
        if not pending or (body.get("req_id") and body["req_id"] != pending["req_id"]):
            return web.json_response({"ok": False, "error": "No matching permission request"}, status=409)
        runner = bridge.runners.get(t.id)
        allow = body.get("allow")
        if isinstance(allow, str):  # HA templates can hand over "True"/"False"
            allow = allow.strip().lower() in ("true", "1", "yes", "on")
        ok = bool(runner and runner.answer_permission(bool(allow)))
        return web.json_response({"ok": ok, "task": t.public()})

    return r


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[voice-bridge] %(asctime)s %(levelname)s %(message)s")
    bridge = Bridge()
    bridge.http = aiohttp.ClientSession()
    app = web.Application(middlewares=[auth])
    app.add_routes(routes(bridge))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info("listening on :%s, claude at %s", PORT, CLAUDE_PATH)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
