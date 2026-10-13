"""MCP server for the Strands demo: the four Azure backends, and the engine that counts.

The Azure port of mcp_server.py in ~/aws-inference-strands and ~/gcp-inference: the same
tools, the same annotations, one server over every backend. Where the AWS server used boto3
and SSM Run Command, this one runs the az CLI, and Run Command on the VM is az vm run-command,
so it needs nothing beyond what stage/ uses.

    read-only    list_backends      the VM, the Container App and the Foundry deployments
                 backend_health     ask one backend for the capital of France, on the box where there is one
                 backend_logs       the serving log: run-command on the VM, az containerapp logs on ACA
                 launch_log         tail the last launch or teardown log
                 ask_backend        one chat request through backends.chat_once, timed here
                 count_ids          the demo 1 engine tool: exact count, min and max for a predicate
    write        launch_backends    stage/up.sh in the background; starts billing
    destructive  teardown           stage/down.sh in the background; deletes the VM and the app

How each health check and log reaches its backend, with no tunnel open:

    vm            az vm run-command as root on the VM; vLLM's key is read from /etc/cci.env there
    aca           HTTPS to the app's ingress with stage/.vllm-key; az containerapp logs
    foundry, phi  the Foundry OpenAI v1 endpoint with the account key; no logs

Launch and teardown call the stage scripts rather than reimplementing them, so the MCP
path and the documented path launch exactly the same thing. teardown runs down.sh without
--all, so the Foundry deployments and the resource group stay.

ask_backend reaches the backends through the variables in stage/demo.env, which is read at
start-up for any variable the environment does not already set; vm needs the SSH tunnel from
stage/forward.sh.

    python3 mcp_server.py                  # stdio; .mcp.json registers it for Claude Code
    python3 demo1_counting.py --tools mcp  # demo 1 with count_ids served from here
"""

import asyncio
import base64
import json
import logging
import os
import re
import shlex
import subprocess
import sys

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backends
import demo1_counting

NAME = "strands-demo"
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(NAME)

mcp = MCPServer(os.getenv("MCP_SERVER_NAME", NAME))
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

# The same defaults as stage/up.sh, forward.sh and down.sh.
RG = os.getenv("RG", "cci-demo-rg")
VM_NAME = "cci-demo-vm"
ACA_APP = os.getenv("ACA_APP", "cci-demo-vllm")
FOUNDRY_RG = os.getenv("FOUNDRY_RG", "research-mesh-rg")
FOUNDRY = os.getenv("FOUNDRY", "research-mesh-foundry")
FOUNDRY_DEPLOYMENTS = {"cci-gpt41-nano": "foundry", "cci-phi4-mini": "phi"}
MANAGED_BY = "cci-demo-run"  # the ManagedBy tag stage/up.sh puts on the VM and the app
REPACK = os.getenv("REPACK", "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4")
KEYFILE = os.path.join(HERE, "stage", ".vllm-key")

HOSTED_BACKENDS = ["foundry", "phi"]   # per token, nothing to launch
LAUNCHED = ["vm", "aca"]
ALL_BACKENDS = HOSTED_BACKENDS + LAUNCHED
HEALTH_QUESTION = "What is the capital of France? Answer in one word."

# Runs on the VM as root: POST one chat request to vLLM on localhost:8000 and print
# "<http status> <seconds>" then the body. Sent base64-encoded so the run-command script is one line.
PROBE = r"""
import json, sys, time, urllib.error, urllib.request
model, key, question = sys.argv[1:4]
body = json.dumps({"model": model, "max_tokens": 16, "temperature": 0.0,
                   "messages": [{"role": "user", "content": question}]}).encode()
req = urllib.request.Request("http://localhost:8000/v1/chat/completions", body,
                             {"content-type": "application/json", "authorization": f"Bearer {key}"})
t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        code, out = r.status, r.read().decode()
except urllib.error.HTTPError as e:
    code, out = e.code, e.read().decode()
except OSError as e:
    code, out = 0, str(e)
print(code, round(time.perf_counter() - t0, 3))
print(out)
"""


def _load_demo_env() -> None:
    """Take stage/demo.env's values for anything the environment leaves unset."""
    path = os.path.join(HERE, "stage", "demo.env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            for key, value in re.findall(r"(\w+)=(\S+)", line.removeprefix("export ")):
                default = re.fullmatch(r"\$\{\w+:-(.*)\}", value)
                os.environ.setdefault(key, default.group(1) if default else value)


_load_demo_env()


class CommandError(RuntimeError):
    """An az command that exited non-zero."""


def _run(*args: str, timeout: int = 180) -> str:
    # stdin is the MCP pipe; a child that inherits it (az does) blocks the server reading it
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False,
                          stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        detail = [x for x in (proc.stderr or proc.stdout).strip().splitlines() if not x.startswith("WARNING")]
        raise CommandError(f"{' '.join(args[:3])} exited {proc.returncode}: {' '.join(detail[-3:])[:600]}")
    return proc.stdout


async def _az(*args: str, timeout: int = 180) -> str:
    return await asyncio.to_thread(_run, "az", *args, "--only-show-errors", timeout=timeout)


async def _json(*args: str) -> list | dict:
    out = await _az(*args, "-o", "json")
    return json.loads(out) if out.strip() else []


def _error(exc: Exception) -> str:
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"❌ timed out after {exc.timeout}s: {' '.join(exc.cmd[:4])}"
    return f"❌ {type(exc).__name__}: {exc}"


def _check_backend(backend: str, allowed: list[str]) -> None:
    if backend not in allowed:
        raise ValueError(f"backend must be one of {', '.join(allowed)}")


def _ours(resource: dict) -> bool:
    return (resource.get("tags") or {}).get("ManagedBy") == MANAGED_BY


# --- discovery: one function per kind of resource, each returning rows for list_backends ---

async def _rg_exists() -> bool:
    return (await _az("group", "exists", "-n", RG)).strip() == "true"


async def _vm() -> list[dict]:
    if not await _rg_exists():
        return []
    rows = await _json("vm", "list", "-d", "-g", RG)
    return [{"backend": "vm", "name": v["name"], "where": v["location"], "type": v["hardwareProfile"]["vmSize"],
             "state": v.get("powerState") or v.get("provisioningState", "?"), "created": v.get("timeCreated", "?"),
             "ip": v.get("publicIps") or None,
             # a deallocated VM bills for its disk only; "VM stopped" without deallocation still bills
             "billing": v.get("powerState") not in {"VM deallocated", None}}
            for v in rows if _ours(v)]


async def _aca() -> list[dict]:
    if not await _rg_exists():
        return []
    rows = await _json("containerapp", "list", "-g", RG)
    out = []
    for a in rows:
        if not _ours(a):
            continue
        props = a.get("properties", {})
        minimum = props.get("template", {}).get("scale", {}).get("minReplicas", 0) or 0
        out.append({"backend": "aca", "name": a["name"], "where": a["location"],
                    "type": f"workload profile {props.get('workloadProfileName', '?')}",
                    "state": f"{props.get('provisioningState', '?')}, {props.get('runningStatus', '?')}",
                    "min_replicas": minimum, "url": f"https://{props['configuration']['ingress']['fqdn']}/v1"
                    if (props.get("configuration") or {}).get("ingress") else None,
                    "billing": minimum > 0})
    return out


async def _foundry() -> list[dict]:
    rows = await _json("cognitiveservices", "account", "deployment", "list", "-g", FOUNDRY_RG, "-n", FOUNDRY)
    return [{"backend": FOUNDRY_DEPLOYMENTS[d["name"]], "name": d["name"], "where": f"{FOUNDRY} ({FOUNDRY_RG})",
             "type": f"{d['properties']['model']['name']}, {d['sku']['name']}",
             "state": d["properties"].get("provisioningState", "?"), "billing": "per token"}
            for d in rows if d["name"] in FOUNDRY_DEPLOYMENTS]


DISCOVER = {"vm": _vm, "aca": _aca, "foundry": _foundry}


async def _find(backend: str) -> dict:
    """The one live resource behind a backend, or an error naming where it looked."""
    rows = [r for r in await DISCOVER["foundry" if backend in HOSTED_BACKENDS else backend]()
            if r["backend"] == backend]
    if backend == "vm":
        rows = [r for r in rows if r["state"] == "VM running"]
    elif backend == "aca":
        rows = [r for r in rows if r["url"]]
    if not rows:
        where = f"{FOUNDRY} in {FOUNDRY_RG}" if backend in HOSTED_BACKENDS else f"resource group {RG}"
        raise RuntimeError(f"no live {backend} resource (tag ManagedBy={MANAGED_BY}) in {where}")
    return rows[0]


async def _run_command(script: str) -> str:
    """Run a one-line shell script as root on the VM with az vm run-command; return its stdout and stderr."""
    out = await _json("vm", "run-command", "invoke", "-g", RG, "-n", VM_NAME, "--command-id", "RunShellScript",
                      "--scripts", script)
    message = out["value"][0]["message"]
    match = re.search(r"\[stdout\]\n(.*?)\n\[stderr\]\n(.*)", message, re.DOTALL)
    if not match:
        return message
    stdout, stderr = match.group(1).rstrip(), match.group(2).strip()
    return stdout + (f"\n[stderr]\n{stderr}" if stderr else "")


def _vllm_key() -> str:
    if os.getenv("VLLM_KEY"):
        return os.environ["VLLM_KEY"]
    with open(KEYFILE) as f:
        return f.read().strip()


async def _hosted(backend: str, row: dict) -> backends.Backend:
    """A Backend for chat_once: from the environment, else built from what discovery found."""
    found = {b.name: b for b in backends.configured()}
    if backend in found and backend in HOSTED_BACKENDS:
        return found[backend]
    if backend == "aca":
        return backends.Backend("aca", "openai", "vLLM on Container Apps (T4)", REPACK, url=row["url"],
                                key=_vllm_key())
    key = os.getenv("FOUNDRY_KEY") or (await _az("cognitiveservices", "account", "keys", "list", "-g", FOUNDRY_RG,
                                                 "-n", FOUNDRY, "--query", "key1", "-o", "tsv")).strip()
    label = {"foundry": "Azure AI Foundry (gpt-4.1-nano)", "phi": "Azure AI Foundry (Phi-4-mini)"}[backend]
    return backends.Backend(backend, "openai", label, row["name"],
                            url=f"https://{FOUNDRY}.services.ai.azure.com/openai/v1", key=key)


@mcp.tool(title="List demo backends", annotations=READ_ONLY)
async def list_backends() -> str:
    """Every backend the demo uses and its state right now.

    Lists the VM and the Container App tagged ManagedBy=cci-demo-run in resource group cci-demo-rg,
    with power state and replica floor, and the two Foundry deployments. Reports how many are
    billing by the hour now and which backends ask_backend is configured for (from stage/demo.env,
    so stale once the tunnel closes). Counts are computed here.
    """
    try:
        found = await asyncio.gather(*[f() for f in DISCOVER.values()], return_exceptions=True)
        rows: list[dict] = []
        errors = {}
        for name, result in zip(DISCOVER, found):
            if isinstance(result, Exception):
                errors[name] = _error(result)
            else:
                rows += result
        seen = {r["backend"] for r in rows}
        rows += [{"backend": b, "state": "not launched" if b in LAUNCHED else "not deployed", "billing": False}
                 for b in ALL_BACKENDS if b not in seen and not (b in HOSTED_BACKENDS and "foundry" in errors)
                 and b not in errors]
        return json.dumps({"backends": rows,
                           "billing_by_the_hour": sum(r["billing"] is True for r in rows),
                           "lookup_errors": errors,
                           # from the environment and stage/demo.env; stale once the tunnel closes
                           "ask_backend_configured": [b.name for b in backends.configured()]}, indent=2)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check one backend's health", annotations=READ_ONLY)
async def backend_health(backend: str) -> str:
    """Ask one backend for the capital of France and pass it only if the answer says Paris.

    For vm the request runs on the VM itself with az vm run-command, so no tunnel is needed.
    The Container App is found by tag and called at its ingress; foundry and phi through the
    Foundry endpoint. A non-empty answer is not a pass: a broken deploy on this lineage has
    answered with degenerate repetition.

    Args:
        backend: one of foundry, phi, vm, aca
    """
    try:
        _check_backend(backend, ALL_BACKENDS)
        row = await _find(backend)
        if backend == "vm":
            probe = base64.b64encode(PROBE.encode()).decode()
            script = (f". /etc/cci.env && echo {probe} | base64 -d | "
                      f'python3 - "$REPACK" "$VLLM_KEY" {shlex.quote(HEALTH_QUESTION)}')
            out = await _run_command(script)
            status_line, _, payload = out.strip().partition("\n")
            code, seconds = status_line.split()[0], float(status_line.split()[1])
            if code != "200":
                return f"❌ vm: HTTP {code} from port 8000 on {row['name']}: {payload[:300]}"
            text = json.loads(payload.split("\n[stderr]")[0])["choices"][0]["message"]["content"] or ""
            where = f"on {row['name']} ({row['where']}) via run-command"
        else:
            r = await asyncio.to_thread(backends.chat_once, await _hosted(backend, row), HEALTH_QUESTION, 16)
            text, seconds = r["text"], r["seconds"]
            where = f"{row['name']} ({row['where']}), direct over HTTPS"
        ok = "paris" in text.lower()
        return f"{'✅' if ok else '❌'} {backend}: answered {text.strip()[:80]!r} in {seconds:.2f}s, {where}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Tail a backend's serving log", annotations=READ_ONLY)
async def backend_logs(backend: str, tail: int = 60) -> str:
    """Tail one backend's vLLM log.

    vm: docker logs on the VM via run-command, falling back to /var/log/cci-setup.log while the
    GRID driver and the container are still being installed. aca: az containerapp logs, console
    first, the system log (image pull, scheduling, quota) when the console has nothing.

    Args:
        backend: one of vm, aca
        tail: lines to return, 1 to 2000 (aca: at most 300)
    """
    try:
        _check_backend(backend, LAUNCHED)
        n = max(1, min(int(tail), 2000))
        row = await _find(backend)
        head = f"{backend} {row['name']} ({row['where']})"
        if backend == "vm":
            out = await _run_command(f"docker logs --tail {n} vllm 2>&1 || (echo '--- vllm container not running "
                                     f"yet; setup log:'; tail -n {n} /var/log/cci-setup.log)")
            return f"{head}\n```\n{out}\n```"
        logs = ["az", "containerapp", "logs", "show", "-g", RG, "-n", row["name"], "--tail", str(min(n, 300)),
                "--format", "text"]
        out = (await _az(*logs[1:])).strip()
        if not out:
            out = "--- no console log; system log:\n" + (await _az(*logs[1:], "--type", "system")).strip()
        return f"{head}\n```\n{out}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Ask a backend", annotations=READ_ONLY)
async def ask_backend(backend: str, prompt: str, max_tokens: int = 100) -> str:
    """One chat request to one backend, timed here, as stage/ask.py does.

    Tokens per second uses the backend's own completion token count over this server's wall
    clock, so it includes network and prefill. vm needs the SSH tunnel (stage/forward.sh);
    the others are reached directly.

    Args:
        backend: one of foundry, phi, vm, aca
        prompt: the question
        max_tokens: completion limit, 1 to 1024
    """
    try:
        found = {b.name: b for b in backends.configured()}
        if backend not in found:
            return f"❌ {backend} is not configured here; configured: {', '.join(found)}. Run stage/forward.sh."
        r = await asyncio.to_thread(backends.chat_once, found[backend], prompt, max(1, min(int(max_tokens), 1024)))
        return json.dumps(r, indent=2)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Count ids", annotations=READ_ONLY)
def count_ids(op: str, value: int) -> dict:
    """Count the ids matching `id <op> value`, computed by the engine.

    Args:
        op: one of >=, >, <=, <, ==
        value: the integer to compare each id against
    """
    return demo1_counting.count(op, value)


def _background(script: str, args: list[str], log: str) -> str:
    path = os.path.join(HERE, "stage", log)
    with open(path, "w") as out:
        proc = subprocess.Popen([os.path.join(HERE, "stage", script), *args], cwd=HERE, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    return f"started stage/{script} {' '.join(args)} (pid {proc.pid}); output in stage/{log}, read it with launch_log"


@mcp.tool(title="Launch demo backends", annotations=WRITE)
async def launch_backends(names: list[str] | None = None) -> str:
    """Run stage/up.sh in the background. Billing starts now and runs until teardown.

    A backend that already exists is left alone (an existing Container App is scaled back to one
    replica). The VM reboots once for the GRID driver before vLLM starts; poll list_backends,
    backend_logs and backend_health. foundry and phi need no launch.

    Args:
        names: any of vm, aca; empty launches both
    """
    try:
        names = names or []
        bad = [n for n in names if n not in {"vm", "aca"}]
        if bad:
            return f"❌ unknown backend(s) {bad}; use vm aca"
        return _background("up.sh", names, "up.log")
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Tear down every demo backend", annotations=DESTRUCTIVE)
async def teardown() -> str:
    """Run stage/down.sh in the background: delete the VM with its disk, NIC and public IP, delete
    the Container App, close the tunnel, then list what is left. Keeps the resource group, the
    Container Apps environment, the registry and the Foundry deployments. Read the sweep with
    launch_log("down")."""
    try:
        return _background("down.sh", [], "down.log")
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Read the launch or teardown log", annotations=READ_ONLY)
def launch_log(which: str = "up", tail: int = 60) -> str:
    """Tail stage/up.log or stage/down.log.

    Args:
        which: up or down
        tail: lines to return
    """
    if which not in ("up", "down"):
        return "❌ which must be up or down"
    path = os.path.join(HERE, "stage", f"{which}.log")
    if not os.path.exists(path):
        return f"stage/{which}.log does not exist yet"
    with open(path) as f:
        lines = f.read().splitlines()
    return "\n".join(lines[-max(1, min(int(tail), 2000)):])


if __name__ == "__main__":
    mcp.run()
