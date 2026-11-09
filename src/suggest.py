"""Shared model loading + suggestion generation (used by try_it.py and live.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, STYLE_FILE, format_context, load_config, require_username, system_prompt  # noqa: E402


class Chatterbox:
    def __init__(self, cfg: dict | None = None, use_adapter: bool = True):
        from mlx_lm import load  # imported lazily: slow import

        self.cfg = cfg or load_config()
        self.me = require_username(self.cfg)
        adapter = ROOT / self.cfg.get("adapter_path", "adapters")
        has_adapter = (adapter / "adapters.safetensors").exists()
        if use_adapter and not has_adapter:
            print(f"(no trained adapter in {adapter.name}/ yet; using the plain base model)", file=sys.stderr)
        self.using_adapter = use_adapter and has_adapter
        self.model, self.tokenizer = load(self.cfg["model"], adapter_path=str(adapter) if self.using_adapter else None)

        self.style = None
        if STYLE_FILE.exists():
            s = json.loads(STYLE_FILE.read_text())
            # The base model always gets examples; a trained adapter only if it was trained with them.
            if s.get("use_in_prompt") or not self.using_adapter:
                self.style = s.get("examples")

    def messages_for(self, channel: str, context: list[tuple[str, str]]) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt(self.me, channel, self.style)},
            {"role": "user", "content": format_context(context)},
        ]

    def suggest(self, channel: str, context: list[tuple[str, str]], n: int | None = None) -> list[str]:
        return self.suggest_from_messages(self.messages_for(channel, context), n)

    def suggest_from_messages(self, messages: list[dict], n: int | None = None) -> list[str]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        n = n or self.cfg.get("num_suggestions", 3)
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        sampler = make_sampler(temp=self.cfg.get("temperature", 0.8), top_p=0.95)
        out: list[str] = []
        for _ in range(n * 3):  # extra tries in case of duplicates/empties
            text = ""
            for chunk in stream_generate(self.model, self.tokenizer, prompt,
                                         max_tokens=self.cfg.get("max_tokens", 60), sampler=sampler):
                text += chunk.text
                if "\n" in text:
                    break
            msg = self._clean(text)
            if msg and msg.lower() not in {m.lower() for m in out}:
                out.append(msg)
            if len(out) >= n:
                break
        return out

    def _clean(self, text: str) -> str:
        msg = text.split("\n", 1)[0].strip()
        if msg.lower().startswith(self.me + ":"):
            msg = msg[len(self.me) + 1:].strip()
        return msg
