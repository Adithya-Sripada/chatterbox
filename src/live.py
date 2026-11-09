"""Step 5: watch a channel's live chat and suggest messages you might send.

Connects to Twitch chat anonymously (read-only, no login). Nothing is ever sent.

  python src/live.py somestreamer            # press Enter for suggestions, q + Enter to quit
  python src/live.py somestreamer --auto 20  # also refresh every 20 s while chat is active
  python src/live.py somestreamer --show-chat  # print chat as it arrives
"""
from __future__ import annotations

import argparse
import random
import re
import select
import socket
import ssl
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import is_usable_context, load_config  # noqa: E402

HOST, PORT = "irc.chat.twitch.tv", 6697
PRIVMSG = re.compile(r"^(?:@\S+ )?:(\w+)!\S+ PRIVMSG #\w+ :(.*)$")
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


class ChatReader(threading.Thread):
    """Keeps a rolling buffer of (time, user, text) for one channel, reconnecting as needed."""

    def __init__(self, channel: str, maxlen: int = 50, echo: bool = False):
        super().__init__(daemon=True)
        self.channel = channel.lower().lstrip("#")
        self.buf: deque = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.echo = echo
        self.connected = threading.Event()
        self.received = 0

    def run(self):
        backoff = 1
        while True:
            try:
                self._session()
                backoff = 1
            except (OSError, ssl.SSLError) as e:
                self.connected.clear()
                print(f"{DIM}[chat disconnected: {e}; retrying in {backoff}s]{RESET}", file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _session(self):
        raw = socket.create_connection((HOST, PORT), timeout=360)
        sock = ssl.create_default_context().wrap_socket(raw, server_hostname=HOST)
        nick = f"justinfan{random.randint(10000, 99999)}"
        sock.sendall(f"NICK {nick}\r\nJOIN #{self.channel}\r\n".encode())
        pending = b""
        while True:
            data = sock.recv(4096)
            if not data:
                raise OSError("server closed the connection")
            pending += data
            *lines, pending = pending.split(b"\r\n")
            for line in lines:
                self._handle(sock, line.decode("utf-8", "replace"))

    def _handle(self, sock, line: str):
        if line.startswith("PING"):
            sock.sendall(("PONG" + line[4:] + "\r\n").encode())
        elif " JOIN #" in line and not self.connected.is_set():
            self.connected.set()
        elif (m := PRIVMSG.match(line)):
            user, text = m.group(1).lower(), m.group(2)
            if text.startswith("\x01ACTION ") and text.endswith("\x01"):
                text = text[8:-1]
            with self.lock:
                self.buf.append((time.time(), user, " ".join(text.split())))
                self.received += 1
            if self.echo:
                print(f"{DIM}{user}: {text}{RESET}")

    def snapshot(self, cfg: dict) -> list[tuple[str, str]]:
        now = time.time()
        with self.lock:
            items = list(self.buf)
        ctx = [(u, t) for ts, u, t in items
               if now - ts <= cfg["context_seconds"] and is_usable_context(u, t, cfg)]
        return ctx[-cfg["context_messages"]:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("channel")
    ap.add_argument("--auto", type=int, metavar="SECONDS", help="refresh suggestions every N s when chat moved")
    ap.add_argument("--show-chat", action="store_true", help="print chat as it arrives")
    ap.add_argument("--no-adapter", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    reader = ChatReader(args.channel, echo=args.show_chat)
    reader.start()

    print("Loading model...", file=sys.stderr)
    from suggest import Chatterbox
    bot = Chatterbox(cfg, use_adapter=not args.no_adapter)

    if not reader.connected.wait(15):
        print("Still connecting to chat (is the channel name right?)...", file=sys.stderr)
    else:
        print(f"Connected to #{reader.channel} (read-only).")
    print("Press Enter for suggestions, q + Enter to quit." +
          (f" Auto-refresh every {args.auto}s." if args.auto else ""))

    last_count = 0
    while True:
        timeout = args.auto if args.auto else None
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            line = sys.stdin.readline()
            if not line or line.strip().lower() in ("q", "quit", "exit"):
                break
        elif reader.received == last_count:  # auto mode, chat hasn't moved
            continue
        last_count = reader.received

        ctx = reader.snapshot(cfg)
        print(f"\n{DIM}--- recent chat in #{reader.channel} ---")
        for u, t in ctx or [("", "(quiet)")]:
            print(f"{u}: {t}" if u else t)
        print(RESET, end="")
        t0 = time.time()
        for i, s in enumerate(bot.suggest(reader.channel, ctx), 1):
            print(f"  {BOLD}{i}) {s}{RESET}")
        print(f"{DIM}({time.time() - t0:.1f}s){RESET}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
