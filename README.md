# exp05 student pack: G1 shop greeter (True Alpha Co-Build, Phase 2)

A Unitree G1 humanoid stands at a shop entrance and greets customers with its upper body. It can:
1. wai and introduce itself (motion learned from a video);
2. introduce the shop and answer questions, typed or spoken (speech recognition), with talking gestures;
3. point at booths with an open palm.

A vision module (`vision/`) sees the people around the robot: their distance, walking speed, whether they face
the robot, and anyone too close to the arms.

It speaks Thai and English with an offline voice. Everything here runs **in simulation on your own laptop**.
The real robot is driven by the same code, but only together with the staff (see `TESTING_ROBOT.md`).

**Open `STUDENT_GUIDE.html` in your browser.** It has today's agenda, a section per session with things to try
and questions to discuss, the system's interfaces, and troubleshooting.

## Requirements

| | Minimum | Full (with the offline LLM) |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04, x86_64 (Windows: WSL2 is untested) | same |
| Disk | ~2 GB (`--no-ai`), ~11 GB (`--no-llm`) | ~19 GB |
| GPU | none (voice and speech recognition run on the CPU, more slowly) | NVIDIA GPU with ≥ 6 GB VRAM, 8 GB recommended (the AI service uses ~5 GB: TTS + ASR + LLM) |
| Internet | needed for the installation | same |
| Packages | `sudo apt install curl pulseaudio-utils libgl1 libegl1` | same |

## Install

```bash
unzip exp05_student_pack.zip && cd exp05
./install.sh              # full. Or: ./install.sh --no-llm  /  ./install.sh --no-ai
```

The zip holds only the code and small data files. `install.sh` downloads the large files:
- `uv`, if you don't have it;
- the Python 3.10 environments `g1/.venv`, `vision/.venv` and `ai/.venv`;
- the vision models (44 MB, into `vision/models`);
- the AI models from Hugging Face, into `ai/models/hf`.

The installer finishes with a few checks. You can run it again at any time: steps that are already done are skipped.
Add `--video` to also download the example wai video (YouTube, research use only).

## Run

```bash
./run_sim.sh              # MuJoCo window + AI service + operator page at http://localhost:8100
HEADLESS=1 ./run_sim.sh   # without the MuJoCo window
NO_AI=1 ./run_sim.sh      # without voice / Q&A
NO_ASR=1 ./run_sim.sh     # without speech recognition (saves ~1.7 GB GPU)
```

Vision (2.3), in a second terminal while `run_sim.sh` runs:

```bash
cd vision && ./run_vision.sh      # sensor sim (walking mannequins, lidar, camera) + vision on http://localhost:8107
./run_vision.sh webcam            # your laptop webcam instead (no lidar)
```

Stop with **Ctrl+C** in the same terminal. The AI service needs ~40 s to load: wait until the header pill says
`AI TTS+LLM` (or `AI TTS`).

## Files

| Path | What |
|---|---|
| `STUDENT_GUIDE.html` | the guide for today (open it in a browser) |
| `install.sh`, `run_sim.sh`, `run_robot.sh` | install / simulation / real robot (staff only) |
| `g1/deploy/` | the robot server (`server.py`, port 8100), motion, pointing, behaviour, speech, Q&A, sim robot, web page |
| `g1/tools/` | checks (`sim_check`, `server_check`, `point_check`, `behaviour_check`, `listen_check`), model builder, wai clip maker |
| `g1/model/`, `g1/third_party/` | MuJoCo G1 models (Dex3-1 / Inspire hands), Unitree SDK and meshes |
| `ai/service.py` | AI service (port 8105): Thai/English TTS, speech recognition (`/asr`), LLM; `ai/tools/`: video → wai / keep-alive |
| `g1/deploy/listen.py` | speech recognition in the robot server (2.2): microphone, VAD (`g1/third_party/silero_vad`), push-to-talk or always listening; check `g1/tools/listen_check.py` |
| `vision/` | people detection (2.3): `run_vision.sh`, `vision/*.py`, `tools/vision_check.py`, `README.md` (interface, camera notes) |
| `research/`, `ASR_LIBRARIES.md` | research notes for 2.3 and 2.2 |
| `clips/` | the learned wai motion per hand model |
| `shops/mockup/` | the 36 × 30 m mockup shop (map, booths, FAQ) |
| `PLAN.md`, `DEV_README.md`, `TESTING_ROBOT.md` | design and decisions, developer notes, real-robot test procedure |

Third-party code and licences: `g1/third_party/THIRD_PARTY.md`. Example video sources: `videos/SOURCES.md`.
