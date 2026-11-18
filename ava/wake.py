"""تشخیص کلمه‌ی بیدارباش از روی متنِ (اغلب ناقصِ) ویسپر.

چون مدل سبک ویسپر «آوا» را ممکن است «اوا»، «آبا»، «او وا» و... بشنود،
مطابقت را با کمی خطا (فاصله‌ی ویرایشی ۱) انجام می‌دهیم.
"""

from __future__ import annotations

from .utils import normalize_fa

DEFAULT_WORDS = ["آوا", "اوا", "آبا", "اووا", "او وا", "ava", "آ وا"]


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def looks_hallucinated(text: str) -> bool:
    """توهم ویسپر روی نویز: یک واژه‌ی کوتاه که پشت‌سرهم چند بار تکرار شده."""
    toks = normalize_fa(text).split()
    if len(toks) >= 4 and len(set(toks)) <= 2:
        return True
    # تکرار یک کاراکتر (مثل «دییییی»)
    stripped = "".join(normalize_fa(text).split())
    return bool(stripped) and len(set(stripped)) <= 2 and len(stripped) >= 6


def is_wake(text: str, words: list[str] | None = None) -> bool:
    words = words or DEFAULT_WORDS
    norm = normalize_fa(text)
    if not norm or looks_hallucinated(text):
        return False
    targets = {normalize_fa(w) for w in words if normalize_fa(w)}

    # مطابقت مستقیم زیررشته‌ای
    if any(t in norm for t in targets):
        return True

    # مطابقت تقریبی: فقط وقتی گفته کوتاه است (پنجره‌ی بیدارباش) و واژه با «ا» شروع می‌شود
    tokens = norm.split()
    if len(tokens) > 3:
        return False
    single_targets = [t for t in targets if " " not in t]
    for tok in tokens:
        if not (3 <= len(tok) <= 5) or not tok.startswith("ا"):
            continue
        if any(_lev(tok, t) <= 1 for t in single_targets):
            return True
    return False
