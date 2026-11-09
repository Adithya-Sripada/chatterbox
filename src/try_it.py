"""Step 4: try the model offline.

  python src/try_it.py --channel somestreamer          # paste chat lines, blank line = suggest
  cat chat.txt | python src/try_it.py --channel x      # context from a file
  python src/try_it.py --blind 20                      # can you tell your real message from the model's?
  add --no-adapter to compare with the untrained base model
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED_DIR  # noqa: E402
from suggest import Chatterbox  # noqa: E402


def parse_lines(text: str) -> list[tuple[str, str]]:
    ctx = []
    for line in text.splitlines():
        if ":" in line:
            u, t = line.split(":", 1)
            if u.strip() and t.strip():
                ctx.append((u.strip().lower(), t.strip()))
    return ctx


def blind_test(bot: Chatterbox, n: int):
    rows = [json.loads(l) for l in open(PROCESSED_DIR / "test.jsonl", encoding="utf-8")]
    rng = random.Random()
    rows = rng.sample(rows, min(n, len(rows)))
    correct = 0
    for i, row in enumerate(rows, 1):
        msgs = row["messages"]
        real = msgs[2]["content"]
        fakes = bot.suggest_from_messages(msgs[:2], n=2)
        options = [real] + fakes
        rng.shuffle(options)
        print(f"\n===== {i}/{len(rows)} =====\n{msgs[1]['content']}\n")
        for j, o in enumerate(options, 1):
            print(f"  {j}) {o}")
        guess = input("Which one did you actually write? ").strip()
        ok = guess.isdigit() and 1 <= int(guess) <= len(options) and options[int(guess) - 1] == real
        correct += ok
        print("  ✓ right" if ok else f"  ✗ it was {options.index(real) + 1}")
    chance = 1 / 3
    print(f"\nYou spotted your own message {correct}/{len(rows)} times "
          f"(pure guessing ≈ {chance:.0%}). Closer to guessing = more convincing model.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="unknown")
    ap.add_argument("--blind", type=int, metavar="N", help="blind test on N held-out examples")
    ap.add_argument("--no-adapter", action="store_true", help="use the base model without your adapter")
    args = ap.parse_args()

    bot = Chatterbox(use_adapter=not args.no_adapter)

    if args.blind:
        return blind_test(bot, args.blind)

    if not sys.stdin.isatty():
        for s in bot.suggest(args.channel, parse_lines(sys.stdin.read())):
            print(f"→ {s}")
        return

    print("Paste chat lines as `user: message`. Blank line = get suggestions. Ctrl-D to quit.")
    buf: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip():
            buf.append(line)
            continue
        for s in bot.suggest(args.channel, parse_lines("\n".join(buf))):
            print(f"  → {s}")
        buf = []


if __name__ == "__main__":
    main()
