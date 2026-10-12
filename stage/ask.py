"""Ask one backend one question and show the answer with its timing.

For the live moment in each section of the talk:

    python3 stage/ask.py vm
    python3 stage/ask.py aca "What is the capital of France? Answer in one word."
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import backends  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    name = sys.argv[1]
    # Long enough that tokens per second is a decode rate, not the cost of one round trip.
    question = sys.argv[2] if len(sys.argv) > 2 else "Write a short paragraph about the ocean."
    found = {b.name: b for b in backends.configured()}
    if name not in found:
        print(f"{name} is not configured; configured: {', '.join(found)}")
        return 1
    b = found[name]
    r = backends.chat_once(b, question, max_tokens=100)
    print(f"{b.label}  ({b.model})")
    print(f"Q: {question}\nA: {r['text']}")
    print(f"{r['completion_tokens']} tokens in {r['seconds']} s = {r['tokens_per_second']} tok/s "
          "(timed here, network and prefill included)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
