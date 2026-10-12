"""Demo 2: one Strands agent on Azure AI Foundry that calls every backend as a tool.

The agent gets one tool per configured backend, `ask_<name>(prompt)`, plus
`scoreboard()`, and is asked to put the same question to all of them.

Every number comes from code: each ask_* tool times its own request and reads the
backend's completion token count; scoreboard() sorts and formats the table. The
agent's job is to call the tools and quote the table, not to compute or rank.

    python3 demo2_orchestrator.py
    python3 demo2_orchestrator.py --question "Name the largest planet in one word."
"""

import argparse
import json
import sys

import backends

DEFAULT_QUESTION = "What is the capital of France? Answer in one word."
SYSTEM = ("You coordinate several model backends. When asked, put the user's question to every "
          "backend using its ask_ tool, then call scoreboard and quote its table exactly. "
          "Do not compute, round or re-rank any number yourself.")


def build(results: list, orchestrator: backends.Backend):
    from strands import Agent, tool

    def make_ask(b: backends.Backend):
        def ask(prompt: str) -> dict:
            try:
                r = backends.chat_once(b, prompt)
            except Exception as exc:
                r = {"backend": b.name, "label": b.label, "error": f"{type(exc).__name__}: {exc}"[:200]}
            results.append(r)
            return r
        ask.__name__ = f"ask_{b.name}"
        ask.__doc__ = (f"Ask {b.label} ({b.model}) one question and return its answer with "
                       f"measured seconds and tokens per second.\n\nArgs:\n    prompt: the question")
        return tool(ask)

    @tool
    def scoreboard() -> str:
        """Return the measured results of every ask_ call so far as a table, fastest first."""
        ok = sorted((r for r in results if "error" not in r), key=lambda r: r["seconds"])
        lines = ["| backend | answer | seconds | tokens/s |", "|---|---|---|---|"]
        lines += [f"| {r['label']} | {r['text'][:40]} | {r['seconds']} | {r['tokens_per_second']} |" for r in ok]
        lines += [f"| {r['label']} | ERROR {r['error'][:60]} | - | - |" for r in results if "error" in r]
        return "\n".join(lines)

    tools = [make_ask(b) for b in backends.configured()] + [scoreboard]
    return Agent(model=backends.strands_model(orchestrator), system_prompt=SYSTEM, tools=tools,
                 callback_handler=None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--json", help="write the measured results here")
    a = ap.parse_args()

    found = backends.configured()
    orchestrator = next(b for b in found if b.name == "foundry")
    print(f"Orchestrator: {orchestrator.label} ({orchestrator.model})")
    print(f"Backends as tools: {', '.join(b.name for b in found)}\n")

    results: list = []
    agent = build(results, orchestrator)
    reply = agent(f"Ask every backend this question and show the scoreboard: {a.question}")
    print(str(reply).strip())

    asked = {r["backend"] for r in results}
    missed = [b.name for b in found if b.name not in asked]
    print(f"\n[harness] {len(results)} backend calls made; not asked by the agent: {', '.join(missed) or 'none'}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"question": a.question, "orchestrator": orchestrator.model, "results": results}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
