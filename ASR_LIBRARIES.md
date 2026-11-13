# 2.2 Speech recognition — library research

Research for STUDENT_GUIDE.html §2.2 (2026-11-13).

**Implemented the same day** (see the guide §2.2 and `g1/deploy/listen.py`): Typhoon Whisper Turbo in the AI service
(transformers, fp16, ~1.7 GB GPU, 0.3–0.6 s per question; with language "th" it also writes English words and English
sentences in English, so one model serves both), Silero VAD v6 through onnxruntime on the CPU, Off / Push-to-talk /
Auto-listen, half-duplex mute, the relevance and presence gates described below. Not done yet: CTranslate2 / faster-whisper
conversion, phrase boosting, echo cancellation / barge-in, wake word, a real-voice test set.
All accuracy numbers come from the model authors (vendor numbers), measured on different test sets. **They are not comparable
across rows.** Our own Thai/English recordings decide.

## What exp05 needs

| Need | Consequence |
|---|---|
| Customers ask short questions (2–8 s), Thai mostly, some English, and Thai sentences with English words in them ("iPhone", "case") | Recognise one utterance at a time, after VAD or push-to-talk. Streaming is optional: the answer can't start before the question ends anyway |
| Offline, on the laptop's RTX 4060 8 GB. exp05 already uses ~4.3 GB (TTS + Typhoon LLM + F5) | ASR must use ≲ 2 GB VRAM, or run on the CPU |
| Answer under ~2 s in total (guide §2.1) | ASR ≲ 0.5 s for a 5 s question |
| Booth and product names | Hotword / phrase boosting helps, or fix the names after ASR (fuzzy match against shop.json) |
| The robot must not hear itself | Mute the mic while `/api/live` reports speech (half-duplex). Echo cancellation is a later step |
| Thailand's PDPA | Process audio in memory. Store nothing by default |

## ASR models

| Model | Size / licence | Runtime | Mode | Thai accuracy (vendor) | Notes for us |
|---|---|---|---|---|---|
| **Typhoon Whisper Turbo** `typhoon-ai/typhoon-whisper-turbo` | 0.8B, MIT (+ OpenTyphoon terms) | `transformers` (the ai env already has 4.55) | offline | TVSpeech 6.85 / GigaSpeech2 4.79 / FLEURS 10.52 CER (Typhoon ASR Benchmark, Jan 2026) | **First choice.** Best Thai CER of the models that fit. ~1.6 GB fp16. Convert to CTranslate2 (`ct2-transformers-converter`) for faster-whisper int8 at ~1 GB and lower latency. Fine-tuned for Thai: measure English and Thai+English mixes |
| Typhoon Whisper Large-v3 | 1.55B, MIT | transformers | offline | 6.32 / 4.69 / 9.98 CER | Only slightly better than Turbo, and ~3 GB. Doesn't fit next to the LLM |
| **Typhoon ASR Real-time** `typhoon-ai/typhoon-asr-realtime` | 114M, CC-BY-4.0 | `pip install typhoon-asr` (NeMo FastConformer-Transducer) | offline batch, very fast | 9.99 / 6.81 / 13.87 CER | **CPU fallback.** Runs on the CPU, so it uses no GPU memory. Thai only. Simple CLI/Python API |
| Typhoon ASR Streaming 115M `typhoon-ai/typhoon-asr-streaming-115m` | 115M, CC-BY-4.0 | NeMo + github.com/warit-s/typhoon-asr-streaming (pinned commit) | cache-aware streaming, 80–1040 ms chunks | TVSpeech 19.4 CER @1040 ms | **Phrase boosting** + n-gram shallow fusion (booth names). The card advises using it only when English code-switching is light |
| Typhoon ASR Streaming Nemotron 0.6B `typhoon-ai/typhoon-asr-streaming-nemotron-0.6b` | 0.6B, OpenMDW-1.1 | NeMo, language prompt `th-TH` | streaming | TVSpeech 14.1 / GigaSpeech2 9.3 CER @1040 ms | Phrase boosting (keyword recall +17→21 %). Thai and English tags. Heavy NeMo install. Use it for a later "live captions while the customer talks" step |
| Thonburian Whisper (biodatlab) `whisper-th-*-combined`, `distill-whisper-th-medium`, distilled large-v3 turbo | small → large, Apache-2.0 | transformers; community CT2 builds exist | offline | CV13 WER: medium 6.6 %, distill-medium 7.6 %, large 5.5 % | 2024 models. Typhoon's 2026 benchmark puts biodatlab large-v3 at 14.9 CER on TVSpeech. Use as a baseline / second opinion |
| Pathumma Whisper `nectec/Pathumma-whisper-th-large-v3` | 1.55B | transformers | offline | TVSpeech 10.36 / GigaSpeech2 5.84 CER | Too large for our VRAM budget |
| OpenAI Whisper large-v3-turbo (+ faster-whisper CT2 builds) | 0.8B, MIT | faster-whisper / whisper.cpp | offline | weaker on Thai than the Thai fine-tunes | Best for **English** and automatic language detection. Could run beside a Thai model, or route by the page's TH/EN switch |
| Qwen3-ASR-0.6B `Qwen/Qwen3-ASR-0.6B` | 0.9B, Apache-2.0 | `pip install qwen-asr` (streaming needs vLLM) | offline + streaming | Thai FLEURS **WER** 10.37 (Whisper-large-v3: 6.85) | Multilingual, with language ID. Weaker on Thai. Worth one comparison run for mixed Thai/English |
| sherpa-onnx `sherpa-onnx-zipformer-thai-2024-06-20` (GigaSpeech2) | small, Apache-2.0 | `pip install sherpa-onnx` (ONNX, CPU) | offline | not checked | Smallest, fastest CPU option. Its VAD + ASR demo is a good template. Could also run on the robot's PC2 |

**Recommended test order:** (1) Typhoon Whisper Turbo through transformers, then CT2 int8 → (2) `typhoon-asr` on the CPU → (3) Whisper large-v3-turbo for English
→ (4) Qwen3-ASR-0.6B → (5) the NeMo streaming models, only if streaming or boosting turns out to matter.

## Runtimes / wrappers

| Library | Use |
|---|---|
| **faster-whisper** (CTranslate2) | Fastest Whisper on NVIDIA, int8, low VRAM. Has a built-in Silero VAD filter. Needs a CT2 conversion of the Typhoon model |
| HF `transformers` pipeline | Simplest first test, with no conversion. Slower |
| whisper.cpp | C/C++, CUDA/Vulkan/CPU, built-in VAD. Choose it if we ever need ASR on the robot's onboard PC |
| RealtimeSTT | High-level mic → VAD → faster-whisper loop (wraps faster-whisper + Silero/WebRTC VAD). Quick prototype of the live-mic step |
| NVIDIA NeMo | Needed for all Typhoon FastConformer/Nemotron models except the `typhoon-asr` pip package. Large dependency: install it in its own env |
| sherpa-onnx | ONNX runtime for Zipformer/Paraformer/Whisper plus VAD and keyword spotting. CPU-friendly |

## Voice activity detection

| Option | Licence | Verdict |
|---|---|---|
| **Silero VAD v6** | MIT, ~2 MB, 16 kHz | **Default.** Best in noise (ESC-50 noise accuracy 0.87, vs 0.61 for v5) |
| TEN VAD | Apache-2.0 **with extra conditions** (restricts third-party use) | Check the licence before any shop deployment |
| WebRTC VAD | BSD | Only for quiet rooms. Poor in noise |

Tune `min_silence` (~600–800 ms, so Thai sentence-final particles like "ครับ/ค่ะ" are not cut off) and a minimum speech length (to drop clicks).

## Triggering, self-hearing, echo

- **Phase 1: push-to-talk** (an operator key or a page button). No wake word needed.
- **Phase 2: VAD while a person is in front** (from 2.3). Mute the mic from `/api/live` speaking=true until ~300 ms after it ends (half-duplex).
- **Barge-in** (customer interrupts) needs echo cancellation with the robot's audio as reference. Options: `pywebrtc-audio` / `aec-audio-processing`
  (WebRTC APM: AEC + noise suppression + AGC), `speexaec` (SpeexDSP). The hard part is the reference alignment, especially when audio plays
  through the robot's speaker over the SDK. Postpone this.
- **Wake word** (optional): `livekit-wakeword` (based on openWakeWord, lists Thai, trains from synthetic TTS) or openWakeWord. Thai wake words are known
  to be weak. Test with several speakers.

## Always listening (no push-to-talk)

Possible and cheap: Silero VAD runs on the CPU all the time, and ASR runs only on the speech segments it finds. The hard part
is the shop entrance, where the robot must not answer everything it hears.

| Problem | Mitigation |
|---|---|
| Self-hearing | Mute the mic while `/api/live` reports speech, plus ~300 ms after it ends. Add AEC later, only if barge-in is needed |
| Passers-by, staff chatter, music | Listen only when someone is engaged (gates below). Use a directional mic aimed at the customer spot and a minimum loudness threshold |
| Whisper hallucinations on noise | Whisper models invent phrases on silence or noise (Thai example: "ขอบคุณที่รับชม"). Drop results with high `no_speech_prob`, too-short speech, or known phantom phrases |
| Speech not meant for the robot | Relevance filter: answer only if the text matches a FAQ, a booth or a shop topic. Otherwise ignore it, or answer "ขอโทษค่ะ ไม่แน่ใจ…" only when a person is clearly in front |
| Privacy (PDPA) | Continuous listening in public needs a visible notice. Process audio in memory and store nothing |

**When to listen (best first):**
1. **A customer is present** (from 2.3): a person within ~1.2 m for more than 2 s, facing the robot. This matches IDLE → GREET → LISTEN in the guide's state machine. It is the most reliable way to avoid answering passers-by.
2. **Wake word** (the robot's name). Customers won't know it, and Thai wake words are weak, so use it as a backup only.
3. **Follow-up window**: keep listening for ~8 s after each answer, so follow-up questions need no trigger.

**Recommended:** always listening, gated by customer presence, with half-duplex mute and the relevance filter. Keep push-to-talk
as an operator override. Until 2.3 exists, test in the sim with an "Auto-listen" switch on the page that stands in for "customer present".

## Microphones

- **Start:** a USB / headset mic on the laptop, then a directional (cardioid / shotgun) mic aimed at the customer position at 1–1.5 m.
- **G1 built-in mic array** (not tested by us):
  - Raw audio: an open unitree_sdk2_python PR #133 (Jan 2026) records the G1 EDU mic from UDP multicast **239.168.123.161:5555, 16 kHz mono PCM**
    on the PC side. Users report it only streams when the robot's voice assistant is in "Wake-up Conversation Mode" (Unitree app → Device → Data → Voice Assistant), and one user still timed out.
  - Unitree's own offline ASR publishes recognised text on DDS `rt/audio_msg` (String) in wake-up mode. Its language support is unverified; likely not Thai.
    The doc mirror could not be loaded. Check Unitree's official G1 "VuiClient / audio" developer docs.
  - These are on the robot side. Follow the real-robot rules (TESTING_ROBOT.md) before trying them.

## Proposed module shape (for the integration discussion)

Separate env `exp05/asr/` (uv, own venv like `ai/`), HTTP service on **:8106**:
`mic → Silero VAD → ASR → {text, lang, confidence, audio_ms, asr_ms}` → client calls `POST :8100/api/ask` → `/api/say` or `/api/point`.
Fuzzy-match booth names from the active shop.json on the text before `/api/ask` (cheap substitute for phrase boosting).

## Test plan (what to measure)

1. Test set: 50–100 questions from the FAQ/booths, ≥ 5 speakers, 3 conditions (quiet, shop noise, 1.5 m). Consent forms. Keep the files only on the lab disk.
2. Metrics: Thai CER (pythainlp-normalised), English WER, booth-name hit rate, latency from end of speech to text, VRAM peak **with the AI service loaded**.
3. End-to-end: does `/api/ask` pick the right booth/answer from the ASR text, compared with the typed text?

## Sources

- Typhoon ASR Real-time paper (arXiv 2601.13044) + benchmark: https://arxiv.org/abs/2601.13044, https://opentyphoon.ai/blog/en/typhoon-asr-realtime-release
- Model cards: https://huggingface.co/typhoon-ai/typhoon-whisper-turbo, https://huggingface.co/typhoon-ai/typhoon-whisper-large-v3,
  https://huggingface.co/typhoon-ai/typhoon-asr-realtime, https://huggingface.co/typhoon-ai/typhoon-asr-streaming-115m,
  https://huggingface.co/typhoon-ai/typhoon-asr-streaming-nemotron-0.6b, https://huggingface.co/typhoon-ai/typhoon-whisper-medium-ctc
- Typhoon ASR Streaming (shallow fusion) paper: https://arxiv.org/pdf/2609.14991
- Thonburian Whisper: https://aclanthology.org/2024.icnlsp-1.17.pdf, https://huggingface.co/biodatlab/distill-whisper-th-medium
- Qwen3-ASR: https://huggingface.co/Qwen/Qwen3-ASR-0.6B, https://arxiv.org/abs/2601.21337
- sherpa-onnx: https://k2-fsa.github.io/sherpa/onnx/pretrained_models
- faster-whisper vs whisper.cpp: https://codersera.com/blog/faster-whisper-vs-whisper-cpp-speech-to-text-2026/
- VAD: https://github.com/TEN-framework/ten-vad, https://www.cekura.ai/blogs/what-is-silero-vad
- AEC: https://strandsagents.com/docs/labs/pywebrtc-audio/, https://pypi.org/project/aec-audio-processing/, https://pypi.org/project/speexaec/
- Wake word: https://github.com/livekit/livekit-wakeword
- G1 mic: https://github.com/unitreerobotics/unitree_sdk2_python/pull/133
