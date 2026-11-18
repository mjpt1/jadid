"""دانلود صدای فارسی Piper و (در صورت وجود) کشیدن مدل Ollama.

اجرا:  python -m scripts.download_models
"""

from __future__ import annotations

import ava._io_utf8  # noqa: F401  — خروجی کنسول UTF-8

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

VOICE_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx"
)
FILES = {
    "models/piper/fa_IR-amir-medium.onnx": VOICE_BASE,
    "models/piper/fa_IR-amir-medium.onnx.json": VOICE_BASE + ".json",
}
OLLAMA_MODEL = "qwen2.5:7b"


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  از قبل موجود است: {dest}")
        return
    print(f"  دانلود {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ava-setup"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)
    print(f"  ذخیره شد: {dest}  ({dest.stat().st_size / 1e6:.1f} مگابایت)")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    print("۱) صدای فارسی Piper:")
    for rel, url in FILES.items():
        try:
            _download(url, root / rel)
        except Exception as exc:  # noqa: BLE001
            print(f"  خطا: {exc}", file=sys.stderr)
            return 1

    print("\n۲) مدل زبانی Ollama:")
    if shutil.which("ollama"):
        try:
            subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"  کشیدن مدل ناموفق بود: {exc}", file=sys.stderr)
    else:
        print("  Ollama نصب نیست. از ollama.com نصب کنید و سپس اجرا کنید:")
        print(f"      ollama pull {OLLAMA_MODEL}")

    print("\nآماده است. حالا اجرا کنید:  python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
