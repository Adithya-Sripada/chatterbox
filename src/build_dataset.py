"""Step 2: turn episodes.jsonl into train/valid/test files for MLX LoRA training.

Filters out your commands, link-only messages and spam repeats, drops bot lines from
the context, trims context to the configured window, splits by time and prints stats.

Usage:  python src/build_dataset.py [--show 10]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PROCESSED_DIR, STYLE_FILE, URL_RE, format_context,  # noqa: E402
                    is_usable_context, load_config, require_username, system_prompt)

BATCH_SIZE = 4
SMALL_DATASET = 1500   # below this, add style examples to the prompt and train ~1 epoch
MAX_ITERS = 1500
WORD_RE = re.compile(r"\S+")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", type=int, default=0, help="print N random training examples to spot-check")
    args = ap.parse_args()

    cfg = load_config()
    me = require_username(cfg)
    src = PROCESSED_DIR / "episodes.jsonl"
    if not src.exists():
        raise SystemExit("Run `python src/parse_logs.py` first.")

    episodes = [json.loads(l) for l in open(src, encoding="utf-8")]
    episodes.sort(key=lambda e: e["ts"])

    dropped = Counter()
    seen = set()
    last_said: dict[tuple[str, str], float] = {}
    kept = []
    ctx_n, ctx_s = cfg["context_messages"], cfg["context_seconds"]
    repeat_s = cfg.get("repeat_window_seconds", 30)

    for e in episodes:
        text, chan, ts = e["text"].strip(), e["channel"], e["ts"]
        key = (chan, int(ts), text)
        if key in seen:                       # same line from two overlapping log sources
            dropped["duplicate log line"] += 1
            continue
        seen.add(key)
        if text.startswith("!"):
            dropped["command (!…)"] += 1
            continue
        if not URL_RE.sub("", text).strip():
            dropped["link only"] += 1
            continue
        prev = last_said.get((chan, text.lower()))
        last_said[(chan, text.lower())] = ts
        if prev is not None and ts - prev <= repeat_s:
            dropped["repeat/spam"] += 1
            continue

        ctx = [(u, t) for u, t, ago in e["context"]
               if ago <= ctx_s and is_usable_context(u, t, cfg)][-ctx_n:]
        kept.append({"ts": ts, "channel": chan, "text": text, "context": ctx})

    n = len(kept)
    if n < 40:
        raise SystemExit(f"Only {n} usable messages of yours after filtering; need at least 40 to train. "
                         "Add more logs to data/raw/.")

    # Time-ordered split: oldest 90% train, then valid, newest test.
    n_eval = max(BATCH_SIZE, round(n * 0.05))
    train, valid, test = kept[: n - 2 * n_eval], kept[n - 2 * n_eval: n - n_eval], kept[n - n_eval:]

    rng = random.Random(0)
    small = len(train) < SMALL_DATASET
    pool = sorted({e["text"] for e in train if 3 <= len(e["text"]) <= 80})
    style = rng.sample(pool, min(10, len(pool)))
    STYLE_FILE.write_text(json.dumps({"use_in_prompt": small, "examples": style}, indent=2, ensure_ascii=False))

    def to_chat(e):
        return {"messages": [
            {"role": "system", "content": system_prompt(me, e["channel"], style if small else None)},
            {"role": "user", "content": format_context(e["context"])},
            {"role": "assistant", "content": e["text"]},
        ]}

    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        with open(PROCESSED_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(to_chat(e), ensure_ascii=False) + "\n")

    epoch = math.ceil(len(train) / BATCH_SIZE)
    iters = max(100, epoch) if small else min(MAX_ITERS, round(1.5 * epoch))
    meta = {"username": me, "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
            "batch_size": BATCH_SIZE, "recommended_iters": iters, "style_in_prompt": small}
    (PROCESSED_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- report
    total = len(episodes)
    print(f"Your messages found: {total:,}")
    for reason, c in dropped.most_common():
        print(f"  dropped {c:>7,}  {reason}")
    print(f"Kept: {n:,}  →  train {len(train):,} / valid {len(valid):,} / test {len(test):,}")
    lens = [len(e["text"]) for e in kept]
    print(f"Median message length: {statistics.median(lens):.0f} chars")
    with_ctx = sum(1 for e in kept if e["context"])
    print(f"Messages with chat context before them: {with_ctx / n:.0%}")
    print("\nPer channel:")
    for chan, c in Counter(e["channel"] for e in kept).most_common(10):
        print(f"  {c:>7,}  {chan}")
    words = Counter(w for e in kept for w in WORD_RE.findall(e["text"]))
    print("\nYour most used words/emotes:")
    print("  " + ", ".join(f"{w} ({c})" for w, c in words.most_common(25)))
    print(f"\nRecommended training: {iters} iterations"
          + (" (small dataset: ~1 epoch, style examples added to prompt)" if small else ""))
    print("Next: ./train.sh")

    if args.show:
        print(f"\n--- {args.show} random examples ---")
        for e in rng.sample(train, min(args.show, len(train))):
            print(f"\n[{e['channel']}]\n{format_context(e['context'])}\n  → {me}: {e['text']}")


if __name__ == "__main__":
    main()
