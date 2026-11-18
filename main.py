"""نقطه‌ی ورود دستیار صوتی «آوا».

نمونه‌ها:
    python main.py                 # اجرای کامل صوتی (بیدارباش + گفتار)
    python main.py --text          # حالت متنی برای تست بدون میکروفون
    python main.py --list-devices  # فهرست دستگاه‌های صوتی
    python main.py --forget        # پاک کردن حافظه‌ی ذخیره‌شده
"""

from __future__ import annotations

import argparse
import sys

import ava._io_utf8  # noqa: F401  — خروجی کنسول را UTF-8 می‌کند (باید اول باشد)
from ava.config import load_config
from ava.display import p as print  # noqa: A001  — چاپِ درستِ فارسی


def main() -> int:
    parser = argparse.ArgumentParser(description="دستیار صوتی لوکال آوا")
    parser.add_argument("--config", default="config.yaml", help="مسیر فایل پیکربندی")
    parser.add_argument("--gui", action="store_true", help="رابط تصویری")
    parser.add_argument("--cli", action="store_true", help="حالت ترمینال (پیش‌فرض همین است)")
    parser.add_argument("--text", action="store_true", help="حالت متنی بدون میکروفون")
    parser.add_argument("--list-devices", action="store_true", help="نمایش دستگاه‌های صوتی")
    parser.add_argument("--forget", action="store_true", help="پاک کردن کل حافظه")
    args = parser.parse_args()

    cfg = load_config(args.config)

    from ava.display import set_enabled

    set_enabled(getattr(cfg, "terminal_reshape", True))

    if args.list_devices:
        from ava.audio_io import list_devices

        print(list_devices())
        return 0

    if args.forget:
        from ava.memory import Memory

        Memory(cfg.memory.path).forget_all()
        print("حافظه پاک شد.")
        return 0

    if args.gui:
        from ava.gui import launch

        launch(cfg)
        return 0

    from ava.assistant import Assistant

    try:
        assistant = Assistant(cfg) if not args.text else _text_assistant(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"\nخطا در راه‌اندازی: {exc}\n", file=sys.stderr)
        return 1

    if args.text:
        assistant.run_text()
    else:
        assistant.run()
    return 0


def _text_assistant(cfg):
    """نسخه‌ی سبک بدون بارگذاری میکروفون/ویسپر/piper برای حالت متنی."""
    from ava.assistant import Assistant

    # فقط اجزای لازم را می‌سازیم
    obj = Assistant.__new__(Assistant)
    obj.cfg = cfg
    from ava.llm import LLM
    from ava.memory import Memory

    obj.memory = Memory(cfg.memory.path)
    obj.llm = LLM(
        cfg.llm.host, cfg.llm.model, cfg.llm.temperature, cfg.llm.num_ctx,
        cfg.llm.timeout, cfg.llm.keep_alive, cfg.llm.num_predict,
    )
    obj.history = []
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
