"""نمایش درستِ فارسی در ترمینال‌هایی که BiDi ندارند (مثل کنسول ویندوز).

متن فارسی را به شکل نمایشیِ راست‌به‌چپ تبدیل می‌کند تا وارونه دیده نشود.
"""

from __future__ import annotations

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    _OK = True
except Exception:  # pragma: no cover
    _OK = False


_ENABLED = True


def set_enabled(value: bool) -> None:
    """اگر ترمینالت خودش فارسی را درست نشان می‌دهد (مثل بعضی ترمینال‌های جدید)، خاموشش کن."""
    global _ENABLED
    _ENABLED = bool(value)


def fa(text: str) -> str:
    if not _ENABLED or not _OK or not text:
        return text
    try:
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return text


def p(*parts: object, **kwargs) -> None:
    """مثل print ولی متن فارسی را درست نمایش می‌دهد."""
    sep = kwargs.pop("sep", " ")
    print(sep.join(fa(str(x)) for x in parts), **kwargs)
