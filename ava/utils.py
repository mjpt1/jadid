"""توابع کمکی مشترک."""

from __future__ import annotations

import re

_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")


def normalize_fa(text: str) -> str:
    """یکدست‌سازی متن فارسی برای مقایسه: حذف اعراب، یکسان‌سازی ی/ک، حذف نویسه‌های اضافی."""
    if not text:
        return ""
    text = text.strip().lower()
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ﻩ", "ه").replace("ۀ", "ه")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه")
    text = re.sub(r"[‌‏‎]", " ", text)  # نیم‌فاصله و نویسه‌های جهت
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any(haystack: str, needles: list[str]) -> bool:
    h = normalize_fa(haystack)
    return any(normalize_fa(n) and normalize_fa(n) in h for n in needles)


def clean_for_speech(text: str) -> str:
    """پاک‌سازی متن پیش از تبدیل به گفتار: حذف مارک‌داون، ایموجی و نویسه‌های نامناسب."""
    if not text:
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"[*_#>|]+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # لینک مارک‌داون
    text = re.sub(r"https?://\S+", " ", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()
