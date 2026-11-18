"""تصمیم‌های سبک و بدون مدل: کِی جست‌وجوی وب لازم است، کِی چیزی باید حفظ شود.

روی سخت‌افزار ضعیف، هر فراخوانی مدل چند ثانیه طول می‌کشد؛ این‌ها را با قاعده‌های
ساده تشخیص می‌دهیم تا در هر نوبت فقط یک بار مدل صدا زده شود.
"""

from __future__ import annotations

from .utils import normalize_fa

# --- جست‌وجوی وب ---
_SEARCH_TRIGGERS = [
    "قیمت", "نرخ", "چند شد", "چنده", "چند است", "دلار", "یورو", "طلا", "سکه",
    "بیت کوین", "بیتکوین", "ارز", "بورس", "سهام",
    "امروز", "الان", "همین حالا", "این هفته", "اخیر",
    "خبر", "اخبار", "تیتر",
    "اب و هوا", "هوا امروز", "دما", "بارش", "پیش بینی هوا",
    "نتیجه بازی", "نتیجه مسابقه", "امتیاز بازی",
    "ساعت چند", "تاریخ امروز", "چندم است",
    "اخرین نسخه", "جدیدترین",
]


def _matches(norm: str, tokens: set[str], trigger: str) -> bool:
    trig = normalize_fa(trigger)
    if " " in trig:                       # عبارت چندکلمه‌ای: زیررشته کافی است
        return trig in norm
    return trig in tokens                 # تک‌کلمه: باید دقیقاً یک واژه باشد (نه بخشی از واژه)


def wants_web_search(user_text: str) -> str | None:
    """اگر پرسش به داده‌ی زنده نیاز دارد، عبارت جست‌وجو را برمی‌گرداند، وگرنه None."""
    norm = normalize_fa(user_text)
    tokens = set(norm.split())
    if any(_matches(norm, tokens, t) for t in _SEARCH_TRIGGERS):
        return user_text.strip()
    return None


# --- حافظه ---
_MEMORY_TRIGGERS = [
    "یادت باشه", "یادت باشد", "به خاطر بسپار", "به خاطر داشته باش", "به یاد داشته باش",
    "فراموش نکن", "از این به بعد", "از الان به بعد", "از این پس", "همیشه",
    "صدا بزن", "صدام کن", "خطاب کن", "اسم من", "اسمم", "منو", "من رو", "من را",
    "دوست دارم که", "ترجیح میدم", "لطفا همیشه", "قانون",
]


def looks_like_teaching(user_text: str) -> bool:
    norm = normalize_fa(user_text)
    tokens = set(norm.split())
    return any(_matches(norm, tokens, t) for t in _MEMORY_TRIGGERS)
