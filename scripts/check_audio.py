"""بررسی زنجیره‌ی صدا: میکروفون → بافر → ویسپر → Piper → بلندگو.

اجرا:  python -m scripts.check_audio
با شمارش معکوس ضبط می‌کند، سطح صدا را زنده نشان می‌دهد، رونویسی می‌کند و یک جمله می‌گوید.
"""

from __future__ import annotations

import ava._io_utf8  # noqa: F401
from ava.display import p as print  # noqa: A001

import time

import numpy as np

from ava.audio_io import Mic, rms
from ava.config import load_config
from ava.stt import STT
from ava.tts import TTS

RECORD_SECONDS = 6


def main() -> int:
    cfg = load_config()

    print("۱) باز کردن میکروفون…")
    mic = Mic(cfg.audio.sample_rate, cfg.audio.input_device, RECORD_SECONDS + 1, cfg.audio.input_gain)
    mic.start()
    time.sleep(0.3)

    for c in (3, 2, 1):
        print(f"   ... {c}")
        time.sleep(1)
    print(f"   حالا حرف بزنید ({RECORD_SECONDS} ثانیه):")

    peak = 0.0
    for _ in range(RECORD_SECONDS * 2):
        time.sleep(0.5)
        level = rms(mic.snapshot(0.5))
        peak = max(peak, level)
        bar = "█" * min(int(level / 200), 40)
        print(f"   {int(level):>6}  {bar}")

    audio = mic.snapshot(RECORD_SECONDS)
    mic.stop()

    abs_audio = np.abs(audio.astype(np.int32))
    clip_ratio = float(np.mean(abs_audio >= 32000))
    overall_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    print(f"\n   نمونه‌ها: {audio.size} | بیشینه: {peak:.0f} | RMS کلی: {overall_rms:.0f}"
          f" | نسبت کلیپ‌شدگی: {clip_ratio * 100:.1f}%")

    if clip_ratio > 0.02:
        print("   ⚠ صدا «کلیپ» شده (بیش از حد بلند و اعوجاج‌دار).")
        print("     در تنظیمات صدای ویندوز: Microphone Boost را خاموش کن یا کم کن،")
        print("     یا در config.yaml مقدار audio.input_gain را کمتر کن.")
        _amp_hint(peak)
        return 1
    if peak < 200:
        print("   ⚠ صدای شما به میکروفون نرسید.")
        print("     - ویندوز: Privacy & security → Microphone را روشن کن")
        print("     - سطح میکروفون را در تنظیمات صدای ویندوز بالا ببر")
        _try_all_devices(cfg)
        _amp_hint(peak)
        return 1
    _amp_hint(peak)

    print("۲) رونویسی با ویسپر…")
    stt = STT(cfg.stt.model, cfg.stt.device, cfg.stt.compute_type, cfg.language)
    text = stt.transcribe(audio)
    print(f"   متن شنیده‌شده: «{text or '(چیزی تشخیص داده نشد)'}»")

    print("۳) پخش صدا با Piper…")
    tts = TTS(cfg.tts.model_path, cfg.tts.length_scale, cfg.audio.output_device)
    tts.say("سلام، این یک آزمایش صدا برای دستیار آوا است.")
    print("   اگر این جمله را شنیدید، خروجی صدا سالم است.")
    print("\nزنجیره‌ی صدا بررسی شد.")
    return 0


def _amp_hint(peak: float):
    """بر اساس بیشینه‌ی صدا، مقدار مناسب audio.input_gain را پیشنهاد می‌دهد."""
    if peak < 1:
        return
    ideal = 12000.0 / peak  # هدف: بیشینه حدود ۱۲۰۰۰
    ideal = max(1.0, min(ideal, 20.0))
    if 0.8 < ideal < 1.3:
        print("   سطح صدا خوب است. نیازی به تغییر input_gain نیست.")
    else:
        print(f"   پیشنهاد: در config.yaml مقدار  audio.input_gain: {ideal:.1f}  را بگذار.")


def _try_all_devices(cfg):
    """هر دستگاه ورودی را دو ثانیه امتحان می‌کند تا ببیند کدام صدا می‌گیرد."""
    import sounddevice as sd

    print("\n   آزمایش خودکار همه‌ی میکروفون‌ها — لطفاً پیوسته حرف بزنید…")
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        try:
            m = Mic(cfg.audio.sample_rate, idx, 3, cfg.audio.input_gain)
            m.start()
            time.sleep(2)
            level = rms(m.snapshot(2))
            m.stop()
            flag = "  ← این یکی صدا می‌گیرد!" if level > 200 else ""
            print(f"     [{idx}] {dev['name'][:45]:45}  RMS={int(level):>6}{flag}")
        except Exception as exc:  # noqa: BLE001
            print(f"     [{idx}] {dev['name'][:45]:45}  خطا: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
