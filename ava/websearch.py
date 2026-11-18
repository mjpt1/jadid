"""جست‌وجوی وب بدون کلید API (داک‌داک‌گو)."""

from __future__ import annotations

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - سازگاری با نسخه‌ی قدیمی
    from duckduckgo_search import DDGS  # type: ignore


def search(query: str, max_results: int = 5, region: str = "wt-wt") -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
    try:
        with DDGS(timeout=8) as ddgs:
            return list(ddgs.text(query, region=region, max_results=max_results))
    except Exception as exc:  # noqa: BLE001
        return [{"title": "خطا در جست‌وجو", "href": "", "body": str(exc)}]


def format_context(results: list[dict]) -> str:
    if not results:
        return "نتیجه‌ای از جست‌وجوی وب به دست نیامد."
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        parts.append(f"[{i}] {title}\n{body}")
    return "نتایج جست‌وجوی وب:\n\n" + "\n\n".join(parts)
