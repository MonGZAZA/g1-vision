"""Voice cloning for uploaded reference voices: F5-TTS-THAI (VIZINTZOR, MIT code; model fine-tuned for Thai).

Thai text is spoken in the uploaded voice. The model is Thai-only, so English text returns None and the service
uses the default English voice (Kokoro). Loaded on first use (~0.7 GB GPU). Reference: voices/<name>/ref.wav
(24 kHz mono, <= 12 s, 0.5 s silence at the end) + ref.txt (its transcript), written by the upload endpoint.
"""

import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


class CloneEngine:
    STEPS, CFG, SPEED = 32, 2.0, 0.95  # README: slightly slower speech reduces skipped words

    def __init__(self):
        import f5_tts_th.tts  # noqa: F401  (import check only; the model loads on first use)
        self.tts = None
        self.lock = threading.Lock()

    def _load(self):
        if self.tts is None:
            from f5_tts_th.tts import TTS
            self.tts = TTS(model="v1", hf_cache_dir=str(HERE / "models/hf/f5"))
        return self.tts

    def say(self, text, lang, ref_dir, speed=1.0):
        if lang != "th":
            return None
        ref_dir = Path(ref_dir)
        with self.lock:
            w = self._load().infer(ref_audio=str(ref_dir / "ref.wav"), ref_text=(ref_dir / "ref.txt").read_text(encoding="utf-8"),
                                   gen_text=text, step=self.STEPS, cfg=self.CFG, speed=self.SPEED * speed)
        return np.asarray(w, np.float32)
