"""ورودی/خروجی صدا: میکروفون پیوسته با بافر چرخشی numpy + پخش صدا."""

from __future__ import annotations

import queue
import threading
import time
import wave

import numpy as np
import sounddevice as sd

FRAME_MS = 30  # هر فریم ۳۰ میلی‌ثانیه — سازگار با webrtcvad


def list_devices() -> str:
    return str(sd.query_devices())


class Mic:
    """میکروفون همیشه‌روشن.

    - چند ثانیه‌ی اخیر را در یک بافر چرخشی numpy نگه می‌دارد (برای کلمه‌ی بیدارباش).
    - هنگام ضبط گفتار کاربر، فریم‌ها را در یک صف جداگانه هم می‌ریزد.

    نوشتن در بافر مستقیماً داخل callback انجام می‌شود تا هیچ فریمی گم نشود.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        input_device=None,
        rolling_seconds: float = 5.0,
        gain: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.gain = max(1.0, float(gain))
        self.blocksize = int(sample_rate * FRAME_MS / 1000)
        self._ring_len = max(int(sample_rate * rolling_seconds), self.blocksize * 4)
        self._ring = np.zeros(self._ring_len, dtype=np.int16)
        self._widx = 0
        self._filled = 0
        self._lock = threading.Lock()
        self._cap_q: queue.Queue = queue.Queue()
        self._capturing = False
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.blocksize,
            device=input_device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        x = indata[:, 0]
        if self.gain > 1.0:
            x = np.clip(x.astype(np.float32) * self.gain, -32768, 32767).astype(np.int16)
        n = x.shape[0]
        with self._lock:
            end = self._widx + n
            if end <= self._ring_len:
                self._ring[self._widx:end] = x
            else:
                first = self._ring_len - self._widx
                self._ring[self._widx:] = x[:first]
                self._ring[: n - first] = x[first:]
            self._widx = end % self._ring_len
            self._filled = min(self._filled + n, self._ring_len)
            if self._capturing:
                self._cap_q.put(x.copy())

    def start(self):
        self._stream.start()

    def stop(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def clear(self):
        """بافر چرخشی را خالی می‌کند (بعد از حرف‌زدن آوا، تا پژواک بلندگو شنیده نشود)."""
        with self._lock:
            self._filled = 0
            self._widx = 0

    def snapshot(self, seconds: float) -> np.ndarray:
        want = int(self.sample_rate * seconds)
        with self._lock:
            k = min(want, self._filled)
            start = (self._widx - k) % self._ring_len
            if start + k <= self._ring_len:
                return self._ring[start : start + k].copy()
            tail = self._ring[start:].copy()
            head = self._ring[: (start + k) % self._ring_len].copy()
        return np.concatenate([tail, head])

    # --- ضبط گفتار کاربر ---
    def start_capture(self):
        with self._lock:
            self._cap_q.queue.clear()
            self._capturing = True

    def read_capture(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._cap_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop_capture(self):
        with self._lock:
            self._capturing = False


def save_wav(path: str, audio: np.ndarray, sample_rate: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def play(audio: np.ndarray, sample_rate: int, output_device=None):
    if audio is None or audio.size == 0:
        return
    sd.play(audio, samplerate=sample_rate, device=output_device)
    sd.wait()


def beep(sample_rate: int = 16000, freq: float = 880.0, ms: int = 140, output_device=None):
    t = np.linspace(0, ms / 1000, int(sample_rate * ms / 1000), endpoint=False)
    tone = (0.25 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    fade = int(0.01 * sample_rate)
    if fade > 0 and tone.size > 2 * fade:
        ramp = np.linspace(0, 1, fade)
        tone[:fade] = (tone[:fade] * ramp).astype(np.int16)
        tone[-fade:] = (tone[-fade:] * ramp[::-1]).astype(np.int16)
    play(tone, sample_rate, output_device)
    time.sleep(0.02)
