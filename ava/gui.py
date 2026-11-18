"""رابط تصویری آوا — بصری‌سازِ تمام‌صفحهٔ سبکِ «جارویس» با Tkinter."""

from __future__ import annotations

import math
import queue
import random
import threading
import tkinter as tk
from tkinter import font as tkfont

import numpy as np

from .assistant import Assistant

BG = "#05080f"
FG = "#dfe8f2"
MUTED = "#6f7d90"
USER = "#57e0ff"
AVA = "#8effc4"
CYAN = "#3fe0ff"
CYAN_DIM = "#123f4e"
AMBER = "#ffca61"

STATE_TEXT = {
    "ready": "آماده",
    "listening": "در حال شنیدن",
    "thinking": "در حال پردازش",
    "speaking": "در حال پاسخ",
    "error": "متوجه نشدم",
    "loading": "بارگذاری…",
}
STATE_ACCENT = {
    "idle": CYAN, "listen": USER, "think": AMBER, "speak": AVA, "error": "#ff8b8b",
}


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(v))) for v in rgb)


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _blend(c1, c2, t):
    a, b = _rgb(c1), _rgb(c2)
    return _hex(a[i] + (b[i] - a[i]) * t for i in range(3))


def _rms(a: np.ndarray) -> float:
    if a is None or a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))


class HUD:
    """بصری‌سازِ دایره‌ایِ سبکِ جارویس که کل بوم را پر می‌کند."""

    def __init__(self, canvas: tk.Canvas):
        self.c = canvas
        self.state = "idle"
        self.level = 0.0
        self.rot = 0.0
        self.phase = 0.0
        self._mic = None

    def bind_mic(self, fn):
        self._mic = fn

    def set_state(self, s):
        self.state = s

    def _arc(self, cx, cy, r, start, extent, color, width=2):
        self.c.create_arc(cx - r, cy - r, cx + r, cy + r, start=start, extent=extent,
                          style="arc", outline=color, width=width, tags="hud")

    def _ring(self, cx, cy, r, color, width=1):
        self.c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=width, tags="hud")

    def _dot(self, x, y, rr, color):
        self.c.create_oval(x - rr, y - rr, x + rr, y + rr, fill=color, width=0, tags="hud")

    def tick(self):
        if self.state == "listen" and self._mic is not None:
            tgt = min(self._mic() / 6500.0, 1.0)
        elif self.state == "think":
            tgt = 0.30 + 0.16 * math.sin(self.phase * 2.2)
        elif self.state == "speak":
            tgt = 0.38 + 0.5 * abs(math.sin(self.phase * 3.1)) * (0.6 + 0.4 * random.random())
        else:
            tgt = 0.07 + 0.05 * math.sin(self.phase * 0.8)
        self.level += (tgt - self.level) * 0.3
        self.rot += {"idle": 0.6, "listen": 1.2, "think": 2.4, "speak": 3.2}.get(self.state, 0.6)
        self.phase += 0.12
        self._draw()

    def _draw(self):
        c = self.c
        c.delete("hud")
        w = max(c.winfo_width(), 10)
        h = max(c.winfo_height(), 10)
        cx, cy = w / 2, h / 2
        R = min(w, h) / 2 - 40
        if R < 30:
            return
        accent = STATE_ACCENT.get(self.state, CYAN)
        lvl = self.level

        # قاب گوشه‌ها
        m, b = 18, 22
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            x = m if sx > 0 else w - m
            y = m if sy > 0 else h - m
            c.create_line(x, y, x + b * sx, y, fill=CYAN_DIM, width=2, tags="hud")
            c.create_line(x, y, x, y + b * sy, fill=CYAN_DIM, width=2, tags="hud")

        # خط‌های نشانه‌رو کم‌رنگ
        faint = _blend(BG, CYAN_DIM, 0.7)
        c.create_line(cx - R, cy, cx + R, cy, fill=faint, tags="hud")
        c.create_line(cx, cy - R, cx, cy + R, fill=faint, tags="hud")

        # حلقه‌ی بیرونی + ۷۲ تیک
        self._ring(cx, cy, R, CYAN_DIM, 1)
        for i in range(72):
            a = math.radians(i * 5)
            big = i % 6 == 0
            r2 = R - (14 if big else 7)
            col = _blend(CYAN_DIM, accent, 0.55 if big else 0.18)
            c.create_line(cx + R * math.cos(a), cy + R * math.sin(a),
                          cx + r2 * math.cos(a), cy + r2 * math.sin(a), fill=col, tags="hud")

        # کمان‌های چرخان
        for k in range(3):
            self._arc(cx, cy, R - 26, self.rot + k * 120, 52, _blend(BG, accent, 0.85), 3)
        for k in range(4):
            self._arc(cx, cy, R - 50, -self.rot * 1.5 + k * 90, 28, _blend(BG, accent, 0.5), 2)
        for k in range(2):
            self._arc(cx, cy, R - 72, self.rot * 0.8 + k * 180, 70, _blend(BG, accent, 0.35), 2)

        # حلقه‌ی نقطه‌ایِ واکنش‌گر
        rseg = R - 96
        if rseg > 10:
            for i in range(32):
                a = math.radians(i * (360 / 32) + self.rot * 0.4)
                wave = 0.5 + 0.5 * math.sin(self.phase * 2 + i * 0.55)
                bright = min(1.0, 0.15 + wave * 0.45 + lvl * 0.7)
                rr = 2 + 4 * (wave * 0.35 + lvl * 0.9)
                self._dot(cx + rseg * math.cos(a), cy + rseg * math.sin(a), rr,
                          _blend(CYAN_DIM, accent, bright))

        # حلقه‌ی داخلیِ تپنده
        rin = max(12, (R - 130) + lvl * 18)
        self._ring(cx, cy, rin, _blend(BG, accent, 0.75), 2)

        # هسته + هاله‌ها
        core = max(10, (R * 0.10) + lvl * (R * 0.20))
        for i, t in enumerate((0.12, 0.22, 0.4, 0.7)):
            self._dot(cx, cy, core * (3.0 - i * 0.6), _blend(BG, accent, t * (0.3 + lvl * 0.5)))
        self._dot(cx, cy, core, _blend(accent, "#ffffff", 0.25 + lvl * 0.4))


class AvaGUI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.q: queue.Queue = queue.Queue()
        self.assistant: Assistant | None = None
        self.busy = False
        self.state = "loading"
        self.last_user = ""
        self.last_ava = ""
        self.load_msg = "شروع…"

        self.root = tk.Tk()
        self.root.title("آوا")
        self.root.configure(bg=BG)
        self.root.geometry("960x720")
        try:
            self.root.state("zoomed")            # بیشینه در ویندوز
        except tk.TclError:
            pass

        fam = tkfont.nametofont("TkDefaultFont").actual("family")
        self.f_title = (fam, 26, "bold")
        self.f_status = (fam, 15)
        self.f_line = (fam, 15)
        self.f_hint = (fam, 12)

        self.viz = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.viz.pack(fill="both", expand=True)
        self.hud = HUD(self.viz)

        self.viz.bind("<Button-1>", lambda e: self._on_talk())
        self.root.bind("<space>", lambda e: self._on_talk())
        self.root.bind("<F11>", lambda e: self._toggle_fs())
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self._fs = False

        threading.Thread(target=self._load_models, daemon=True).start()
        self.root.after(60, self._drain)
        self._animate()

    def _toggle_fs(self):
        self._fs = not self._fs
        self.root.attributes("-fullscreen", self._fs)

    # ---------- رندر ----------
    def _animate(self):
        self.hud.tick()
        self._draw_overlay()
        self.root.after(40, self._animate)

    def _draw_overlay(self):
        c = self.viz
        c.delete("overlay")
        w = max(c.winfo_width(), 10)
        h = max(c.winfo_height(), 10)
        accent = STATE_ACCENT.get(self.hud.state, CYAN)

        c.create_text(w / 2, 34, text="A V A", fill=CYAN, font=self.f_title, tags="overlay")
        c.create_text(w / 2, 66, text=STATE_TEXT.get(self.state, self.state),
                      fill=accent, font=self.f_status, tags="overlay")

        wrap = min(int(w * 0.8), 900)
        y = h - 96
        if self.last_ava:
            c.create_text(w / 2, y, text="آوا:  " + self.last_ava, fill=AVA,
                          font=self.f_line, width=wrap, justify="center", tags="overlay")
        if self.last_user:
            c.create_text(w / 2, y - 40, text="شما:  " + self.last_user, fill=USER,
                          font=self.f_line, width=wrap, justify="center", tags="overlay")

        if self.state == "loading":
            hint = self.load_msg or "در حال بارگذاری…"
        elif self.busy:
            hint = "…"
        else:
            hint = "روی صفحه کلیک کن یا Space را بزن و صحبت کن"
        c.create_text(w / 2, h - 34, text=hint, fill=MUTED, font=self.f_hint, tags="overlay")

    def _set_state(self, state: str):
        self.state = state
        self.hud.set_state({"listening": "listen", "thinking": "think",
                            "speaking": "speak", "error": "error"}.get(state, "idle"))

    # ---------- بارگذاری ----------
    def _load_models(self):
        try:
            a = Assistant(self.cfg, on_load=lambda m: self.q.put(("loadmsg", m)))
            a.on_status = lambda s, t="": self.q.put(("state", s))
            a.on_user = lambda t: self.q.put(("user", t))
            a.on_answer = lambda t: self.q.put(("ava", t))
            a.mic.start()
            self.hud.bind_mic(lambda: _rms(a.mic.snapshot(0.12)))
            self.assistant = a
            self.q.put(("ready_init",))
            self.q.put(("loadmsg", "گرم‌کردن مدل زبانی (اولین پاسخ کمی طول می‌کشد)…"))
            a.llm.warmup()
            self.q.put(("loadmsg", ""))
        except Exception as exc:  # noqa: BLE001
            self.q.put(("fatal", str(exc)))

    # ---------- تعامل ----------
    def _on_talk(self):
        if self.busy or self.assistant is None:
            return
        self.busy = True
        self._set_state("listening")
        threading.Thread(target=self._run_turn, daemon=True).start()

    def _run_turn(self):
        try:
            self.assistant._handle_turn()
        except Exception as exc:  # noqa: BLE001
            self.q.put(("state", "error"))
            self.q.put(("ava_err", str(exc)))
        finally:
            self.q.put(("turn_done",))

    # ---------- صف ----------
    def _drain(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "loadmsg":
                    self.load_msg = msg[1]
                elif kind == "ready_init":
                    self._set_state("ready")
                elif kind == "fatal":
                    self._set_state("error")
                    self.last_ava = f"خطا در راه‌اندازی: {msg[1]}"
                    self.load_msg = ""
                elif kind == "state":
                    self._set_state(msg[1])
                elif kind == "user":
                    self.last_user, self.last_ava = msg[1], ""
                elif kind == "ava":
                    self.last_ava = msg[1]
                elif kind == "ava_err":
                    self.last_ava = f"خطا: {msg[1]}"
                elif kind == "turn_done":
                    self.busy = False
                    if self.state not in ("error",):
                        self._set_state("ready")
        except queue.Empty:
            pass
        self.root.after(60, self._drain)

    def run(self):
        self.root.mainloop()
        if self.assistant:
            self.assistant.mic.stop()


def launch(cfg):
    AvaGUI(cfg).run()
