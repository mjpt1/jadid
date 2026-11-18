"""تبدیل متن به گفتار فارسی با Piper (آفلاین).

از رابط خط‌فرمان piper استفاده می‌کند که بین نسخه‌ها پایدارترین است.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from . import voice_fx
from .audio_io import play
from .utils import clean_for_speech


class TTSUnavailable(RuntimeError):
    pass


class TTS:
    def __init__(
        self,
        model_path: str,
        length_scale: float = 1.0,
        output_device=None,
        robot: bool = False,
        robot_carrier_hz: float = 42.0,
        robot_mix: float = 0.18,
        robot_bits: int = 9,
    ):
        self.model_path = Path(model_path)
        self.length_scale = length_scale
        self.output_device = output_device
        self.robot = robot
        self.robot_carrier_hz = robot_carrier_hz
        self.robot_mix = robot_mix
        self.robot_bits = robot_bits
        self.exe = shutil.which("piper") or shutil.which("piper.exe")

        if not self.exe:
            raise TTSUnavailable(
                "دستور piper پیدا نشد. بسته‌ی piper-tts را نصب کنید:  pip install piper-tts"
            )
        if not self.model_path.exists():
            raise TTSUnavailable(
                f"فایل صدای فارسی پیدا نشد: {self.model_path}\n"
                "این دستور را اجرا کنید:  python -m scripts.download_models"
            )
        cfg_path = Path(str(self.model_path) + ".json")
        self.sample_rate = 22050
        if cfg_path.exists():
            try:
                self.sample_rate = json.loads(cfg_path.read_text(encoding="utf-8"))["audio"]["sample_rate"]
            except Exception:
                pass

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        text = clean_for_speech(text)
        if not text:
            return np.array([], dtype=np.int16), self.sample_rate
        # کودک piper باید ورودی را UTF-8 بخواند، وگرنه روی ویندوز فارسی خراب می‌شود
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ava.wav"
            txt_in = Path(tmp) / "in.txt"
            txt_in.write_text(text, encoding="utf-8")
            proc = subprocess.run(
                [
                    self.exe,
                    "--model", str(self.model_path),
                    "--output_file", str(out),
                    "--length_scale", str(self.length_scale),
                ],
                stdin=txt_in.open("rb"),
                capture_output=True,
                env=env,
            )
            if proc.returncode != 0 or not out.exists():
                sys.stderr.write(proc.stderr.decode("utf-8", "ignore") + "\n")
                raise TTSUnavailable("اجرای piper با خطا مواجه شد.")
            with wave.open(str(out), "rb") as wf:
                sr = wf.getframerate()
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        if self.robot:
            audio = voice_fx.robotize(
                audio, sr, self.robot_carrier_hz, self.robot_mix, self.robot_bits
            )
        return audio, sr

    def say(self, text: str):
        audio, sr = self.synthesize(text)
        play(audio, sr, self.output_device)
