"""اجبار خروجی کنسول به UTF-8 تا متن فارسی روی ویندوز درست چاپ شود.

این ماژول باید پیش از هر چاپ فارسی import شود.
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


def force_utf8() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


force_utf8()
