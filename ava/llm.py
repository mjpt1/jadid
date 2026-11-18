"""اتصال به مدل زبانی لوکال از طریق Ollama."""

from __future__ import annotations

import json

import requests

SYSTEM_PROMPT = """تو «آوا» هستی؛ یک دستیار صوتیِ لوکال. فقط و فقط به فارسیِ روان و درست پاسخ بده.

قواعد:
- کوتاه و دقیق حرف بزن. معمولاً یک تا سه جمله. هیچ مقدمه و تعارف و توضیحِ اضافه نده.
- جوابِ مستقیمِ همان چیزی را بده که پرسیده شده. اگر سؤال نامفهوم بود، در یک جمله بپرس منظورش چیست.
- درباره‌ی اینکه «هوش مصنوعی هستی» یا «محدودیت داری» توضیح نده مگر مستقیم پرسیده شود.
- اگر پاسخ را نمی‌دانی، صادقانه و کوتاه بگو نمی‌دانی.
- خروجی تو به صدا تبدیل می‌شود: از علامت، فهرست، جدول، ایموجی و انگلیسی پرهیز کن. فقط جملهٔ فارسیِ ساده.
- اگر «نتایج جست‌وجوی وب» به تو داده شد، خلاصه و دقیق از آن استفاده کن.
- نکاتی که کاربر قبلاً یاد داده (بخش حافظه) را همیشه رعایت کن."""

_SEARCH_DECISION = """بر اساس پیام کاربر تصمیم بگیر که آیا برای پاسخ درست به جست‌وجوی زنده‌ی وب نیاز است یا نه.
اگر نیاز است، فقط یک شیء JSON در یک خط برگردان به شکل: {"search": true, "query": "<عبارت جست‌وجوی مناسب به فارسی یا انگلیسی>"}
اگر نیاز نیست: {"search": false}
هیچ متن اضافه‌ای ننویس."""

_PRINCIPLE_EXTRACT = """وظیفه: تشخیص بده آیا کاربر در پیام زیر «واقعیت ماندگار درباره‌ی خودش»، «ترجیح»، «دستور رفتاری» یا «اصلاحِ» یک اشتباه را بیان کرده که باید برای همیشه به خاطر سپرده شود.
اگر بله، آن را در یک جمله‌ی کوتاه فارسی و به صورت دستوری برای دستیار بنویس (مثلاً: «کاربر را آقای رضایی خطاب کن» یا «واحد پول پیش‌فرض تومان است»).
اگر صرفاً یک پرسش یا گفت‌وگوی معمولی است، دقیقاً کلمه‌ی NONE را برگردان.
فقط همان یک جمله یا NONE را بنویس."""


class OllamaError(RuntimeError):
    pass


class LLM:
    def __init__(
        self,
        host: str,
        model: str,
        temperature: float = 0.6,
        num_ctx: int = 3072,
        timeout: int = 600,
        keep_alive: str = "30m",
        num_predict: int = 140,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.num_predict = num_predict

    # --- بررسی سلامت ---
    def check(self) -> None:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(
                f"اتصال به Ollama روی {self.host} برقرار نشد. مطمئن شو سرویس در حال اجراست (ollama serve)."
            ) from exc
        names = {m["name"] for m in resp.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise OllamaError(
                f"مدل «{self.model}» در Ollama نصب نیست. اجرا کن:  ollama pull {self.model}"
            )

    # --- فراخوانی پایه ---
    def _chat(self, messages: list[dict], temperature: float | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"خطا در فراخوانی مدل: {exc}") from exc
        return resp.json().get("message", {}).get("content", "").strip()

    def warmup(self) -> None:
        """یک فراخوانی کوتاه تا مدل در حافظه بار شود و پاسخ‌های بعدی سریع باشند."""
        try:
            requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "سلام"}],
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"num_predict": 1, "num_ctx": self.num_ctx},
                },
                timeout=self.timeout,
            )
        except Exception:  # noqa: BLE001
            pass

    # --- کاربردها ---
    def answer(
        self,
        user_text: str,
        history: list[dict],
        memory_text: str = "",
        web_context: str = "",
    ) -> str:
        system = SYSTEM_PROMPT
        if memory_text:
            system += "\n\n[حافظه]\n" + memory_text
        messages = [{"role": "system", "content": system}]
        messages += history
        content = user_text
        if web_context:
            content = f"{web_context}\n\n---\nپرسش کاربر: {user_text}"
        messages.append({"role": "user", "content": content})
        return self._chat(messages)

    def needs_search(self, user_text: str) -> str | None:
        raw = self._chat(
            [
                {"role": "system", "content": _SEARCH_DECISION},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
        )
        try:
            start, end = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[start : end + 1])
        except Exception:  # noqa: BLE001
            return None
        if data.get("search"):
            return (data.get("query") or user_text).strip()
        return None

    def extract_principle(self, user_text: str) -> str | None:
        raw = self._chat(
            [
                {"role": "system", "content": _PRINCIPLE_EXTRACT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
        ).strip()
        if not raw or raw.upper().startswith("NONE") or len(raw) > 200:
            return None
        return raw
