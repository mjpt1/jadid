"""حلقه‌ی اصلی دستیار «آوا»: بیدارباش، گوش دادن، فهمیدن، پاسخ صوتی."""

from __future__ import annotations

import time

from . import heuristics, voice_fx, wake, websearch
from .audio_io import Mic, play, rms, save_wav
from .display import fa
from .display import p as print  # چاپِ درستِ فارسی در ترمینال
from .llm import LLM
from .memory import Memory
from .recorder import record_utterance
from .stt import STT
from .tts import TTS


class Assistant:
    def __init__(self, cfg, on_load=None):
        self.cfg = cfg
        say = on_load or (lambda msg: print(msg))

        say("اتصال به مدل زبانی…")
        self.memory = Memory(cfg.memory.path)
        self.llm = LLM(
            cfg.llm.host, cfg.llm.model, cfg.llm.temperature, cfg.llm.num_ctx,
            cfg.llm.timeout, cfg.llm.keep_alive, cfg.llm.num_predict,
        )
        self.llm.check()

        say("بارگذاری موتور تشخیص گفتار…")
        self.stt = STT(cfg.stt.model, cfg.stt.device, cfg.stt.compute_type, cfg.language)
        use_wake = getattr(cfg, "interaction", "push_to_talk") == "wake"
        if not use_wake or cfg.wake.model == cfg.stt.model:
            self.wake_stt = self.stt
        else:
            say("بارگذاری مدل کلمه‌ی بیدارباش…")
            self.wake_stt = STT(cfg.wake.model, cfg.stt.device, cfg.stt.compute_type, cfg.language)

        say("بارگذاری موتور گفتار…")
        self.tts = TTS(
            cfg.tts.model_path, cfg.tts.length_scale, cfg.audio.output_device,
            robot=cfg.tts.robot,
            robot_carrier_hz=cfg.tts.robot_carrier_hz,
            robot_mix=cfg.tts.robot_mix,
            robot_bits=cfg.tts.robot_bits,
        )
        self.sr = cfg.audio.sample_rate
        self.fx = cfg.tts.sound_effects

        self.mic = Mic(
            sample_rate=cfg.audio.sample_rate,
            input_device=cfg.audio.input_device,
            rolling_seconds=max(3.0, cfg.wake.window_seconds + 1.0),
            gain=cfg.audio.input_gain,
        )
        self.history: list[dict] = []
        self.listening = False

        # قلاب‌ها برای رابط تصویری (در حالت ترمینال بی‌اثرند)
        self.on_status = lambda state, text="": None   # ready|listening|thinking|speaking|error
        self.on_user = lambda text: None
        self.on_answer = lambda text: None

    def _maybe_search(self, user_text: str) -> str:
        if not self.cfg.websearch.enabled:
            return ""
        query = heuristics.wants_web_search(user_text)
        if not query:
            return ""
        print(f"  جست‌وجوی وب: {query}")
        try:
            results = websearch.search(
                query, self.cfg.websearch.max_results, self.cfg.websearch.region
            )
            return websearch.format_context(results)
        except Exception as exc:  # noqa: BLE001
            print(f"  (هشدار: جست‌وجوی وب ناموفق: {exc})")
            return ""

    # --- مرحله‌ها ---
    def _wait_for_wake(self):
        w = self.cfg.wake
        debug = getattr(w, "debug", False)
        print("\nمنتظر کلمه‌ی بیدارباش… («آوا» را بگویید)")
        while True:
            time.sleep(w.hop_seconds)
            chunk = self.mic.snapshot(w.window_seconds)
            level = rms(chunk)
            if level < w.rms_gate:
                if debug:
                    print(f"    [سکوت] RMS={level:.0f}")
                continue
            text = self.wake_stt.transcribe(chunk, vad_filter=False, permissive=True)
            if debug:
                save_wav("debug_wake.wav", chunk, self.mic.sample_rate)
                print(f"    [شنیده] RMS={level:.0f}  نمونه={chunk.size}  متن=«{text}»")
            if text and wake.is_wake(text, w.words):
                if not debug:
                    print(f"  ← شنیدم: «{text}»")
                return

    def _cue(self, kind: str):
        if not self.fx:
            return
        fn = {"listen": voice_fx.listen_cue, "think": voice_fx.thinking_cue,
              "error": voice_fx.error_cue}[kind]
        play(fn(self.sr), self.sr, self.cfg.audio.output_device)

    def _handle_turn(self):
        self._cue("listen")
        print("بفرمایید، گوش می‌دهم…")
        self.on_status("listening")
        self.listening = True
        audio = record_utterance(
            self.mic,
            aggressiveness=self.cfg.vad.aggressiveness,
            silence_ms=self.cfg.vad.silence_ms,
            max_seconds=self.cfg.vad.max_utterance_seconds,
            start_timeout=10.0,
        )
        self.listening = False
        dbg = getattr(self.cfg, "debug", False)
        if dbg:
            save_wav("debug_turn.wav", audio, self.mic.sample_rate)
            print(f"    [ضبط] نمونه={audio.size}  RMS={rms(audio):.0f}")
        if audio.size < self.mic.sample_rate // 2:  # کمتر از نیم ثانیه = چیزی گفته نشد
            print("چیزی نشنیدم.")
            self._cue("error")
            self.on_status("error", "چیزی نشنیدم")
            return

        self.on_status("thinking")
        user_text = self.stt.transcribe(audio)
        if dbg:
            print(f"    [رونویسی] «{user_text}»")
        if not user_text.strip() or wake.looks_hallucinated(user_text):
            print("متوجه نشدم.")
            self._cue("error")
            self.on_status("error", "متوجه نشدم")
            return
        print(f"کاربر: {user_text}")
        self.on_user(user_text)
        self._cue("think")

        # ۱) یادگیری اصل تازه (فقط وقتی جمله بوی «آموزش» می‌دهد؛ سپس یک بار مدل)
        if heuristics.looks_like_teaching(user_text):
            try:
                principle = self.llm.extract_principle(user_text)
                if principle and self.memory.add(principle):
                    print(f"  ✓ به حافظه اضافه شد: {principle}")
            except Exception as exc:  # noqa: BLE001
                print(f"  (هشدار: استخراج حافظه ناموفق: {exc})")

        # ۲) جست‌وجوی وب (تصمیم با قاعده‌ی سبک، بدون فراخوانی مدل)
        web_context = self._maybe_search(user_text)

        # ۳) پاسخ
        answer = self.llm.answer(
            user_text, self.history, self.memory.render(), web_context
        )
        print(f"آوا: {answer}")
        self.on_answer(answer)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        max_msgs = self.cfg.llm.history_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

        # ۴) گفتار — و پاک‌کردن بافر تا پژواک بلندگو به‌عنوان بیدارباش شنیده نشود
        self.on_status("speaking")
        self.tts.say(answer)
        time.sleep(0.3)
        self.mic.clear()
        self.on_status("ready")

    def run(self):
        self.mic.start()
        print("گرم‌کردن مدل زبانی…")
        self.llm.warmup()
        mode = getattr(self.cfg, "interaction", "push_to_talk")
        self.tts.say("واحد آوا فعال شد.")
        time.sleep(0.3)
        self.mic.clear()
        try:
            if mode == "wake":
                while True:
                    self._wait_for_wake()
                    self._handle_turn()
            else:
                self._run_push_to_talk()
        except KeyboardInterrupt:
            print("\nخداحافظ.")
        finally:
            self.mic.stop()

    def _run_push_to_talk(self):
        print("\nحالت «کلید و صحبت»: کلید Enter را بزن و بعد سؤالت را بپرس.")
        print("برای خروج: Ctrl+C\n")
        while True:
            try:
                input(fa("↵ Enter بزن و حرف بزن… "))
            except EOFError:
                break
            self._handle_turn()

    # --- حالت متنی برای تست بدون میکروفون ---
    def run_text(self):
        self.llm.check()
        print("حالت متنی آوا. برای خروج «خروج» را بنویسید.\n")
        while True:
            try:
                user_text = input(fa("شما: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_text in {"خروج", "exit", "quit"}:
                break
            if not user_text:
                continue
            if heuristics.looks_like_teaching(user_text):
                principle = self.llm.extract_principle(user_text)
                if principle and self.memory.add(principle):
                    print(f"  ✓ به حافظه اضافه شد: {principle}")
            web_context = self._maybe_search(user_text)
            answer = self.llm.answer(user_text, self.history, self.memory.render(), web_context)
            print(f"آوا: {answer}\n")
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": answer})
