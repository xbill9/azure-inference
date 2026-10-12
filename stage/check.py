"""Pre-talk check: is every configured backend answering, and do tool calls work?

For each backend in the environment (source stage/demo.env first) it sends one
short question; for each tool-capable backend it also runs demo 1 once with the
engine tool. Exits 1 if anything failed, so it can gate the start of the demo.

    python3 stage/check.py
    python3 stage/check.py --wait 1800    # keep retrying until all pass or 30 min pass
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import backends  # noqa: E402
import demo1_counting  # noqa: E402

QUESTION = "What is the capital of France? Answer in one word."


def check(b: backends.Backend) -> tuple[bool, str]:
    try:
        r = backends.chat_once(b, QUESTION, max_tokens=16)
    except Exception as exc:
        return False, f"chat failed: {type(exc).__name__}: {exc}"[:160]
    if "paris" not in r["text"].lower():
        return False, f"chat answered {r['text'][:60]!r}"
    note = f"chat {r['seconds']}s"
    if b.tool_capable:
        run = demo1_counting.run_once(b, "engine")
        if run["error"]:
            return False, f"{note}; tool run failed: {run['error'][:120]}"
        if not run["filters"]:
            return False, f"{note}; tool run sent no count_ids call (tool calling off?)"
        note += f"; tool run filters={run['filters']} quoted={run['quoted']} {run['seconds']}s"
    return True, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=int, default=0, help="retry failing backends for up to this many seconds")
    a = ap.parse_args()

    pending = backends.configured()
    print("Configured:", ", ".join(b.name for b in pending))
    deadline = time.time() + a.wait
    failed: dict = {}
    while True:
        still = []
        for b in pending:
            ok, note = check(b)
            stamp = time.strftime("%H:%M:%SZ", time.gmtime())
            print(f"  {stamp}  {'PASS' if ok else 'FAIL'}  {b.name:<10} {note}", flush=True)
            if ok:
                failed.pop(b.name, None)
            else:
                failed[b.name] = note
                still.append(b)
        pending = still
        if not pending or time.time() >= deadline:
            break
        print(f"  ... {len(pending)} not ready, retrying in 60 s", flush=True)
        time.sleep(60)
    missing = [n for n, v in {"foundry": "FOUNDRY_URL", "phi": "PHI_MODEL", "aca": "ACA_URL",
                              "vm": "VM_URL"}.items() if not os.environ.get(v)]
    if missing:
        print("Not configured:", ", ".join(missing))
    print("READY" if not failed else f"NOT READY: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
