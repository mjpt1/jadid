"""حافظه‌ی کاری ماندگار: اصولی که کاربر به آوا یاد می‌دهد."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .utils import normalize_fa


class Memory:
    def __init__(self, path: str):
        self.path = Path(path)
        self.principles: list[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.principles = data.get("principles", [])
            except Exception:
                self.principles = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"principles": self.principles}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        norm = normalize_fa(text)
        for p in self.principles:
            if normalize_fa(p["text"]) == norm:
                return False
        self.principles.append({"text": text, "added_at": datetime.now().isoformat(timespec="seconds")})
        self._save()
        return True

    def forget_all(self):
        self.principles = []
        self._save()

    def render(self) -> str:
        if not self.principles:
            return ""
        lines = [f"- {p['text']}" for p in self.principles]
        return "اصول و نکاتی که کاربر قبلاً به تو یاد داده و باید همیشه رعایت کنی:\n" + "\n".join(lines)
