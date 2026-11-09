"""Shared config loading and chat filtering rules (used by training and live mode alike)."""
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
STYLE_FILE = PROCESSED_DIR / "style_examples.json"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


def load_config() -> dict:
    with open(ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    cfg["bots"] = {b.lower() for b in cfg.get("bots", [])}
    cfg["username"] = (os.environ.get("CHATTERBOX_USERNAME") or cfg.get("username", "")).strip().lstrip("@").lower()
    return cfg


def require_username(cfg: dict) -> str:
    user = cfg["username"]
    if not user or user == "change_me":
        raise SystemExit('Set your Twitch username in config.toml (username = "...") first.')
    return user


def is_usable_context(user: str, text: str, cfg: dict) -> bool:
    """Whether another chatter's line is worth keeping as context."""
    if user.lower() in cfg["bots"]:
        return False
    if not text.strip() or len(text) > cfg.get("max_context_chars", 300):
        return False
    return True


def system_prompt(username: str, channel: str, style_examples: list[str] | None = None) -> str:
    prompt = (
        f"You are {username} chatting in {channel}'s Twitch chat. "
        "Reply with one message in your usual style."
    )
    if style_examples:
        prompt += "\nSome things you've said before:\n" + "\n".join(f"- {m}" for m in style_examples)
    return prompt


def format_context(lines: list[tuple[str, str]]) -> str:
    """lines: [(user, text), ...] oldest first."""
    if not lines:
        return "(chat is quiet)"
    return "\n".join(f"{u}: {t}" for u, t in lines)
