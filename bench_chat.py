"""Same prompt, every backend, a few repeats: wall-clock seconds and tokens per second.

Each request is timed here (chat_once) and its completion token count comes from
the backend, so tokens/s includes the network path and the prefill. The medians
and ranges in the summary are computed here; nothing is estimated.

    python3 bench_chat.py --repeats 5 --json results/bench.json
"""

import argparse
import json
import statistics
import sys

import backends

PROMPT = "Write a short paragraph about the ocean."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--json")
    a = ap.parse_args()

    rows = []
    for b in backends.configured():
        backends.chat_once(b, "Hi", max_tokens=4)  # warm the path; not recorded
        for _ in range(a.repeats):
            r = backends.chat_once(b, PROMPT, max_tokens=a.max_tokens)
            rows.append(r)

    print(f"Prompt: {PROMPT!r}, max_tokens={a.max_tokens}, {a.repeats} repeats per backend\n")
    print("backend     median s  median tok/s  tok/s range     tokens")
    summary = []
    for name in dict.fromkeys(r["backend"] for r in rows):
        mine = [r for r in rows if r["backend"] == name]
        tps = [r["tokens_per_second"] for r in mine]
        secs = [r["seconds"] for r in mine]
        toks = [r["completion_tokens"] for r in mine]
        s = {"backend": name, "label": mine[0]["label"], "median_seconds": round(statistics.median(secs), 3),
             "median_tokens_per_second": round(statistics.median(tps), 1), "min_tps": min(tps), "max_tps": max(tps),
             "tokens": f"{min(toks)}-{max(toks)}"}
        summary.append(s)
        print(f"{name:<10} {s['median_seconds']:>9}  {s['median_tokens_per_second']:>12}  "
              f"{s['min_tps']:>6}-{s['max_tps']:<7}  {s['tokens']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"prompt": PROMPT, "max_tokens": a.max_tokens, "summary": summary, "runs": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
