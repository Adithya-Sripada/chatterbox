"""Step 1: stream every log file in data/raw/ and keep only the moments where YOU chatted.

For each of your messages it writes one "episode" (your message + the chat lines just
before it) to data/processed/episodes.jsonl. Everything else in the logs is discarded,
so multi-GB logs shrink to a small file and are never loaded into memory at once
(except whole-file JSON exports, which have to be parsed in one go).

Supported formats (auto-detected per line or per file):
  - Chatterino:          [12:34:56]  someuser: message        (date from filename/header)
  - justlog / rustlog:   [2024-05-01 12:34:56] #channel someuser: message
  - Dated text exports:  [2024-05-01 12:34:56 UTC] someuser: message
  - Relative VOD text:   [1:02:03] someuser: message
  - Raw IRC:             @tags :someuser!someuser@... PRIVMSG #channel :message
  - JSON / JSONL / CSV   with user / message / timestamp style fields
                         (TwitchDownloader, chat-downloader, rustlog JSON, generic)
  - Any of the above gzipped (.gz)

Usage:  python src/parse_logs.py [--raw DIR]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import sys
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED_DIR, RAW_DIR, load_config, require_username  # noqa: E402

# Keep a generous buffer here; build_dataset.py trims it to the configured size
# after dropping bots, so changing config.toml never requires re-parsing.
BUFFER_LINES = 40
BUFFER_SECONDS = 180

RE_RUSTLOG = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:\s?UTC|Z)?\]\s+#(\w+)\s+([^\s:]+):\s?(.*)$"
)
RE_DATED = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:\s?UTC|Z)?\]\s+(.+?):\s(.*)$"
)
RE_TIMED = re.compile(r"^\[(\d{1,2}):(\d{2}):(\d{2})\]\s+(.+?):\s(.*)$")
RE_IRC = re.compile(r"^(?:@(\S+) )?:(\w+)!\S+ PRIVMSG #(\w+) :(.*)$")
RE_DATE_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
RE_CHATTERINO_HEADER = re.compile(r"^# Start logging at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_PAREN_LOGIN = re.compile(r"\((\w+)\)$")

USER_KEYS = ("username", "user_name", "user", "login", "author", "commenter", "name",
             "display_name", "displayName", "sender", "from", "nick", "chatter")
TEXT_KEYS = ("message", "text", "body", "content", "msg", "message_text")
TS_KEYS = ("timestamp", "created_at", "time", "ts", "sent_at", "date", "datetime", "tmi-sent-ts")
OFFSET_KEYS = ("content_offset_seconds", "time_in_seconds", "offset")
CHANNEL_KEYS = ("channel", "channel_name", "channelName", "room", "broadcaster", "streamer")
LIST_KEYS = ("comments", "messages", "chat", "data", "logs", "items")


# ---------------------------------------------------------------- helpers

def norm_user(raw) -> str:
    """'DisplayName', 'Display (login)', 'Display login', {'name': ..} -> 'login'."""
    if isinstance(raw, dict):
        for k in ("name", "login", "username", "display_name", "displayName"):
            if raw.get(k):
                return norm_user(raw[k])
        return ""
    s = str(raw or "").strip().lstrip("@")
    m = RE_PAREN_LOGIN.search(s)
    if m:
        return m.group(1).lower()
    if " " in s:  # localized display name followed by login
        s = s.split()[-1]
    return s.lower()


def norm_text(raw) -> str:
    if isinstance(raw, dict):
        for k in ("body", "text", "message", "content"):
            if isinstance(raw.get(k), str):
                return norm_text(raw[k])
        return ""
    s = str(raw or "")
    if s.startswith("\x01ACTION ") and s.endswith("\x01"):  # /me
        s = s[8:-1]
    return " ".join(s.split())


def parse_ts(raw) -> float | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
    else:
        s = str(raw).strip()
        try:
            v = float(s)
        except ValueError:
            s = s.replace(" UTC", "+00:00").replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(s)
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
    if v > 1e14:   # microseconds
        return v / 1e6
    if v > 1e11:   # milliseconds
        return v / 1e3
    return v


def first(d: dict, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def open_text(path: Path):
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8-sig", errors="replace")
    return open(path, encoding="utf-8-sig", errors="replace")


def logical_suffix(path: Path) -> str:
    return Path(path.stem).suffix.lower() if path.suffix == ".gz" else path.suffix.lower()


def channel_from_path(path: Path, raw_root: Path) -> str:
    """Chatterino: Channels/<chan>/<chan>-2024-05-01.log ; otherwise filename or folder."""
    stem = Path(path.stem).stem if path.suffix == ".gz" else path.stem
    name = RE_DATE_IN_NAME.sub("", stem).strip("-_ #")
    if name and re.fullmatch(r"\w{3,25}", name):
        return name.lower()
    if path.parent != raw_root:
        return path.parent.name.lower()
    return "unknown"


def base_time_for(path: Path, file_index: int) -> float:
    m = RE_DATE_IN_NAME.search(path.name) or RE_DATE_IN_NAME.search(str(path.parent))
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d, tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    # No date anywhere: give each file its own far-apart synthetic time range so
    # files never interleave. Real ordering inside the file is preserved.
    return 1_000_000_000 + file_index * 10_000_000


# ---------------------------------------------------------------- record sources

def iter_text_lines(path: Path, raw_root: Path, file_index: int, stats: Counter, samples: list):
    """Yield (ts, channel, user, text) from a line-based log."""
    default_chan = channel_from_path(path, raw_root)
    base = base_time_for(path, file_index)
    day_offset = 0.0
    last_tod = -1.0
    with open_text(path) as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            stats["lines"] += 1

            m = RE_CHATTERINO_HEADER.match(line)
            if m:
                dt = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
                base = dt.replace(hour=0, minute=0, second=0).timestamp()
                day_offset, last_tod = 0.0, -1.0
                stats["skipped"] += 1
                continue

            m = RE_IRC.match(line)
            if m:
                tags, user, chan, text = m.groups()
                ts = None
                if tags:
                    tm = re.search(r"tmi-sent-ts=(\d+)", tags)
                    ts = parse_ts(tm.group(1)) if tm else None
                yield ts, chan.lower(), user.lower(), norm_text(text)
                continue

            m = RE_RUSTLOG.match(line)
            if m:
                ts_s, chan, user, text = m.groups()
                yield parse_ts(ts_s), chan.lower(), norm_user(user), norm_text(text)
                continue

            m = RE_DATED.match(line)
            if m:
                ts_s, user, text = m.groups()
                yield parse_ts(ts_s), default_chan, norm_user(user), norm_text(text)
                continue

            m = RE_TIMED.match(line)
            if m:
                h, mi, s, user, text = m.groups()
                tod = int(h) * 3600 + int(mi) * 60 + int(s)
                if tod + 3600 < last_tod:  # clock went backwards: past midnight
                    day_offset += 86400
                last_tod = tod
                yield base + day_offset + tod, default_chan, norm_user(user), norm_text(text)
                continue

            stats["skipped"] += 1
            if len(samples) < 3:
                samples.append(line[:160])


def records_to_rows(records, default_chan: str, base: float, stats: Counter):
    rows = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        stats["lines"] += 1
        user = norm_user(first(rec, USER_KEYS))
        text = norm_text(first(rec, TEXT_KEYS))
        if not user or not text:
            stats["skipped"] += 1
            continue
        ts = parse_ts(first(rec, TS_KEYS))
        if ts is None:
            off = first(rec, OFFSET_KEYS)
            ts = base + float(off) if off is not None else None
        chan = first(rec, CHANNEL_KEYS)
        chan = norm_user(chan) if chan else default_chan
        rows.append((ts, chan, user, text))
    return rows


def iter_structured(path: Path, raw_root: Path, file_index: int, stats: Counter, samples: list):
    default_chan = channel_from_path(path, raw_root)
    base = base_time_for(path, file_index)
    suffix = logical_suffix(path)

    if suffix in (".csv", ".tsv"):
        with open_text(path) as f:
            reader = csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ",")
            yield from records_to_rows(reader, default_chan, base, stats)
        return

    is_jsonl = suffix in (".jsonl", ".ndjson")
    if not is_jsonl and suffix not in (".json",):
        with open_text(path) as f:  # .txt/.log holding one JSON object per line?
            is_jsonl = f.read(1) == "{" and f.readline().rstrip().endswith("}") and bool(f.readline().strip())
    if is_jsonl:
        with open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    stats["lines"] += 1
                    stats["skipped"] += 1
                    if len(samples) < 3:
                        samples.append(line[:160])
                    continue
                yield from records_to_rows([rec], default_chan, base, stats)
        return

    # Whole-file JSON (TwitchDownloader, rustlog JSON, chat-downloader...)
    with open_text(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        streamer = data.get("streamer") or data.get("channel") or (data.get("video") or {}).get("channel")
        if streamer:
            default_chan = norm_user(streamer)
        created = parse_ts((data.get("video") or {}).get("created_at"))
        if created:
            base = created
        records = next((data[k] for k in LIST_KEYS if isinstance(data.get(k), list)), [])
    else:
        records = data
    rows = records_to_rows(records, default_chan, base, stats)
    rows.sort(key=lambda r: (r[0] is None, r[0] or 0))
    yield from rows


def sniff_structured(path: Path) -> bool:
    """True for JSON/JSONL/CSV; False for line-based text logs."""
    suffix = logical_suffix(path)
    if suffix in (".json", ".jsonl", ".ndjson", ".csv", ".tsv"):
        return True
    with open_text(path) as f:
        head = f.read(256).lstrip()
    # Text logs also start with "[" ("[12:34:56] ..."), so require an object after it.
    return head.startswith("{") or re.match(r"\[\s*\{", head) is not None


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DIR, help="folder with your log files")
    args = ap.parse_args()

    cfg = load_config()
    me = require_username(cfg)
    raw_root = args.raw.resolve()
    files = sorted(p for p in raw_root.rglob("*") if p.is_file() and not p.name.startswith("."))
    if not files:
        raise SystemExit(f"No log files found in {raw_root}. Copy your chat logs there first.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "episodes.jsonl"
    totals = Counter()
    users_seen = Counter()

    print(f"Looking for messages from '{me}' in {len(files)} file(s)...\n")
    print(f"{'file':60} {'lines':>9} {'parsed':>9} {'skipped':>8} {'yours':>7}")
    with open(out_path, "w", encoding="utf-8") as out:
        for i, path in enumerate(files):
            stats, samples = Counter(), []
            buffers: dict[str, deque] = {}
            last_ts = 0.0
            try:
                structured = sniff_structured(path)
                source = (iter_structured if structured else iter_text_lines)(path, raw_root, i, stats, samples)
                for ts, chan, user, text in source:
                    if not text or not user:
                        stats["skipped"] += 1
                        continue
                    if ts is None:  # keep order, nudge forward
                        ts = last_ts + 1
                    last_ts = ts
                    stats["parsed"] += 1
                    if len(users_seen) < 200_000:
                        users_seen[user] += 1
                    buf = buffers.setdefault(chan, deque(maxlen=BUFFER_LINES))
                    if user == me:
                        stats["yours"] += 1
                        ctx = [[u, t, round(ts - t0, 1)] for (t0, u, t) in buf if ts - t0 <= BUFFER_SECONDS]
                        out.write(json.dumps({
                            "ts": ts, "channel": chan, "text": text,
                            "context": ctx,  # [user, text, seconds_before]
                            "src": str(path.relative_to(raw_root)),
                        }, ensure_ascii=False) + "\n")
                    buf.append((ts, user, text))
            except (json.JSONDecodeError, UnicodeError, OSError) as e:
                print(f"  ! could not read {path.name}: {e}")
                continue

            rel = str(path.relative_to(raw_root))
            rel = rel if len(rel) <= 60 else "…" + rel[-59:]
            print(f"{rel:60} {stats['lines']:>9,} {stats['parsed']:>9,} {stats['skipped']:>8,} {stats['yours']:>7,}")
            if stats["lines"] and stats["parsed"] / stats["lines"] < 0.5 and samples:
                print("    mostly unrecognized lines here, e.g.:")
                for s in samples:
                    print(f"      {s}")
            totals.update(stats)

    print(f"\nTotal: {totals['lines']:,} lines, {totals['parsed']:,} chat messages, "
          f"{totals['yours']:,} of them yours.")
    if totals["yours"] == 0 and totals["parsed"]:
        close = [u for u, _ in users_seen.most_common() if me[:4] in u][:10]
        print(f"\nNo messages from '{me}' found. Check `username` in config.toml.")
        if close:
            print("Similar names in the logs:", ", ".join(close))
        sys.exit(1)
    print(f"Wrote {out_path.relative_to(PROCESSED_DIR.parent.parent)}. Next: python src/build_dataset.py")


if __name__ == "__main__":
    main()
