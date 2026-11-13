# exp05: G1 shop greeter

The G1 stands at a shop entrance facing into the shop and interacts with its upper body only. It has three tasks:
1. a Thai wai and a self-introduction;
2. a service introduction and typed Q&A;
3. pointing at booths with an open palm.

It speaks with a female Thai/English voice (offline), with captions for testing and a remote monitor page. The
plan, the decisions and the stages are in [`PLAN.md`](PLAN.md). Everything runs on this laptop. It is done and
checked in simulation; the real-robot steps are in [`TESTING_ROBOT.md`](TESTING_ROBOT.md) (S7).

## Status

| Stage | State |
|---|---|
| S0 sim robot + server skeleton | **Done 2026-11-10.** `tools/sim_check.py` 10/10 |
| S1 shop editor + 3D view | **Done 2026-11-10** (API checks in `tools/server_check.py`, 21/21 with S0; page loads without console errors). Mouse editing has not been tested by a person yet |
| S2 pointing | **Done 2026-11-10.** `tools/point_check.py` 15/15: 12 mockup booths + 168 grid targets (±100°, 1–45 m), worst azimuth error 3.5°, near always lower than far, no collision or limit violation. Live in the sim the ray is within 0.5°, ZMP margin ≥ 5.7 cm |
| S3 voice + captions + timeline | **Done 2026-11-10.** AI service (`ai/`): Thai female VITS (MMS-TTS-THAI-FEMALEV2) + English Kokoro af_heart, phrase timing, cache, voice upload + cloning (F5-TTS-THAI, Thai). One clock for motion, audio and captions. Captions start within ±60 ms of the speech (`tools/behaviour_check.py`). Laptop / robot / silent audio. **You pick the voice by listening** |
| S4 wai from video | **Done 2026-11-10** with the example video (MediaPipe instead of GVHMR/HaMeR/GMR, no registration needed): `clips/wai.json` from `videos/examples/x07hnDvNcxM.mp4` 29–36 s, palm gap 1.9 cm, symmetric, within the speed budget. **You judge it next to the video** (Scripts & voice tab). Your own video replaces it (S8). 2026-11-13: user judged the Dex3 wai under standard (palms ~9 cm apart: the Dex3 thumb bases stand ~3 cm proud and touch first); hands now turned 35° inward (`WAI_HAND_TURN_DEG`, gestures.py) so the fingers close in front; old clip in `clips/backup/` |
| S5 keep-alive + gestures, introduce itself / service | **Done 2026-11-10.** Idle layer always on (never frozen: 0 frozen 1-s windows), gestures (welcome, present, both, beats) at phrase starts and `[tags]`, both actions TH/EN end to end. Style from video: tool ready (`ai/tools/video_to_idle.py`); the example presenter videos were too lively, so the defaults stay until your keep-alive video |
| S6 offline Q&A | **Done 2026-11-10.** FAQ match, else Typhoon 2.5 (Qwen3-4B, 4-bit, 2.9 GB GPU, ~1 s) from the shop data only; booth found for "Speak + point". 4/4 test questions right in TH/EN |
| S7 real robot | **Prepared**: `run_robot.sh`, Dex3-1 hand output (`--hands dex3`), robot speaker (PlayStream), latency setting, [`TESTING_ROBOT.md`](TESTING_ROBOT.md). 2026-11-10: the sim has Dex3-1 too (default `HANDS=dex3`); wai/gestures re-solved for Dex3 (thumb folded; rest shoulder roll 0.18). Needs you + the robot |
| S8 your videos + real shop | Tools ready (shop upload/editor, video → wai/idle). Needs your videos and the shop map |
| 2.2 speech recognition | **Done in sim 2026-11-13** (`g1/deploy/listen.py`, AI service `/asr`): mic → Silero VAD v6 (ONNX, CPU) → Typhoon Whisper Turbo (Thai, also writes English; +1.7 GB GPU, ~0.3–0.6 s per question) → filters → Q&A → speak / point. Off / Push-to-talk / Auto-listen, muted while the robot speaks, "only questions" and "only with a customer" (presence from 2.3) filters. `g1/tools/listen_check.py` 18/18. Laptop: the built-in "Digital Microphone" is picked automatically (the default input is the empty headset jack). Robot mic (multicast) written but untested. **You try it by speaking** (Thai TTS voices are a poor test: real voices decide) |
| 2.3 vision: people, facing, distance, speed | **Done in sim 2026-11-13** ([`vision/README.md`](vision/README.md)): camera (MediaPipe pose + YuNet face → facing), head lidar (distance, 360° proximity), Kalman tracks (walking speed), events on :8107 for other modules; own sensor sim (people, lidar, camera). `vision/tools/vision_check.py` 28/28. `cd vision && ./run_vision.sh` after `./run_sim.sh`. Real robot: Unitree's URDF tilts the head D435i 47.6° down (sees legs only) → check a real image or add a level camera |

Checks (all pass 2026-11-10): `sim_check` 10/10, `server_check` 21/21, `point_check` 15/15, `behaviour_check`
17/17. 2026-11-13 (after the wai hand turn): `sim_check` 11/11, `server_check` 22/22, `point_check` 15/15;
`behaviour_check` 16/18: the intro balance margin (ZMP > 2 cm) failed for the old and the new wai clip alike while the
machine was under load (load average 12–21 from other sessions; the sim's ZMP uses finite-difference COM acceleration
and is timing-sensitive). Offline the new raise is gentler (peak ZMP x 4.6 cm vs 5.3 cm). **Re-run on a quiet machine.** Disk: `tools/storage.sh` (limit 200 GB): 16.8 GB (AI env 7 GB, models 9.5 GB).

## Next steps (2026-11-13)

2026-11-13 the user checked the sim: voice OK, gestures OK, Shop tab (mouse) OK, wai under standard → fixed (S4 row).

| Who | Step | Time (estimate) |
|---|---|---|
| User | Look at the new Dex3 wai (Operate → 🙏 Sawasdee + introduce; it is live, no restart needed). Old one: `clips/backup/wai_dex3_before_turn35_2026-11-13.json` | ~5 min |
| Claude | Re-run `behaviour_check` on a quiet machine (intro balance margin) | ~5 min |
| User | Try speech recognition: Operate → Listen → Push-to-talk (hold 🎤 or space) and Auto-listen; ask in Thai and English, e.g. "ห้องน้ำอยู่ที่ไหนคะ", "ร้านเปิดกี่โมง". Report wrong texts / false triggers | ~15 min |
| User + robot + spotter | S7: [`TESTING_ROBOT.md`](TESTING_ROBOT.md), 9 steps, `./run_robot.sh` (Dex3-1 now, `HANDS=inspire` later) | ~2–3 h, + ~1 h fixes |
| User | Record the wai and keep-alive videos to the specs in PLAN.md §7 (full upper body, both hands, tripod) | — |
| Claude | S8: process the user's videos (`video_to_wai.py` / `make_wai_clip.py` / `video_to_idle.py --install`), tune | ~1 h |
| User | Upload the real shop map, place robot and booths (Shop tab), dry run at the entrance | ~0.5–1 h + 1–2 h |

## Run (simulation)

```bash
cd ~/robotics-simulation/exp05 && ./run_sim.sh          # sim (MuJoCo window) + AI service + http://localhost:8100
HEADLESS=1 ./run_sim.sh                                 # no window
NO_LLM=1 ./run_sim.sh                                   # voice without the LLM (saves ~3 GB GPU)
NO_ASR=1 ./run_sim.sh                                   # without speech recognition (saves ~1.7 GB GPU)
HANDS=inspire ./run_sim.sh                              # Inspire 5-finger hands (default dex3 = the robot now)
```

`HANDS` (exported as `EXP05_HANDS`) selects the model for the sim, the server and the motion code. The wai clip
(`clips/wai_<hands>.json`) and the gesture IK cache (`cache/poses_<hands>.json`) are per hand model. The checks
take it from `EXP05_HANDS` (default dex3).

Stop with Ctrl+C in that terminal: the server releases the arms, then the sim and the AI service stop. If it was
started in the background (`nohup ./run_sim.sh &`), Ctrl+C/SIGINT does not reach it: `kill -TERM` the
`deploy/server.py` process first (arms released), then the `sim_robot.py` and `service.py` processes by PID. The AI
service takes ~40 s to load (header pill "AI TTS+LLM+ASR"); it uses ~5.0 GB of GPU with the LLM, the clone voice and
speech recognition (Typhoon Whisper Turbo, downloaded to `ai/models/hf` on the first start, 1.6 GB).
Logs go to `logs/<time>/`. Real robot: `./run_robot.sh` (see TESTING_ROBOT.md).

Checks (each starts its own sim on a private DDS domain and stops only its own processes; `behaviour_check` needs
the AI service running):

```bash
cd g1 && .venv/bin/python tools/sim_check.py && .venv/bin/python tools/server_check.py && .venv/bin/python tools/point_check.py
.venv/bin/python tools/behaviour_check.py [--audio laptop]                         # voice, captions, wai, gestures, Q&A
.venv/bin/python tools/listen_check.py                                              # speech recognition (AI service with ASR)
MUJOCO_GL=egl .venv/bin/python tools/point_render.py /tmp/points.png [--cam front]   # contact sheet of the pointing poses
```

Video → robot motion (S4/S5, S8 with your videos):

```bash
# Speaking speed (0.8-1.2x) and volume (per output: laptop playback level / the robot speaker's own level,
# AudioClient.SetVolume): sliders in the Voice card, saved in cache/audio_settings.json. The speed is applied by the TTS
# engines while generating, so the phrase times (and with them gestures, wai hold, captions) follow; checked by
# g1/tools/behaviour_check.py --rate 0.8 / --rate 1.2.
# Reference voice (cloning), Scripts & voice -> Voice card: pick a reading script (fills the transcript), press
# Record / Stop (browser microphone, saved as WAV; needs http://localhost:8100 on the laptop), listen, Upload. Or upload
# a WAV/MP3 file. Test: g1/tools/record_check.py (Chrome with a fake microphone; needs the AI service).
# From the page (Scripts & voice -> "Teach from a new video"): upload a wai or keep-alive video, optional start/end,
# check the result (palm gap, symmetry, speed, video-vs-robot sheet), then "Use this" (old one -> clips/backup/,
# "Undo" puts it back) or "Keep the old one". Same tools as below, run by deploy/video_jobs.py; uploads in
# videos/uploads/. Test: g1/tools/video_check.py (13 checks). Command line:
cd ai && .venv/bin/python tools/video_to_wai.py VIDEO [--start S --end E]   # -> clips/source/<video>_<t>s_wai.json + sheet
cd ../g1 && .venv/bin/python tools/make_wai_clip.py ../clips/source/<...>_wai.json   # -> clips/wai.json + _compare.jpg
cd ../ai && .venv/bin/python tools/video_to_idle.py VIDEO --install                  # keep-alive style -> clips/idle.json
```

## Operator page (http://localhost:8100)

Header: **ไทย / EN** language, audio output (laptop / robot / off), AI service status, **monitor ↗** (read-only
caption + status page for a remote machine: `http://<laptop>:8100/monitor`), **E-STOP**.

**Operate** tab:
- **Actions**: **🙏 Sawasdee + introduce** (wai, "สวัสดีค่ะ"/"Hello!" in its hold, then the self-introduction with
  gestures), **Introduce service**, **Stop** (stops speaking, arms to rest), keep-alive idle on/off.
- **Answer a question**: type (Thai or English) or click a FAQ → the answer appears (FAQ or LLM, editable) →
  **Speak**, or **Speak + point to <booth>** when the answer names a booth.
- **Listen (speech recognition)**: **Off** (mic closed) / **Push-to-talk** (hold the 🎤 button or the space bar,
  release to recognise) / **Auto-listen** (always listening).
  - The mic is muted while the robot speaks (and from the start of an answer until it has been spoken), so it never
    answers itself and a new question does not interrupt an answer.
  - **reply**: *answer automatically* (Q&A → speak; points at the booth when the answer names one) or *put in the
    question box* (the text goes to "Answer a question", you check it and press Speak).
  - **only questions** (Auto-listen): ignores speech that is not a question or request for the robot (no FAQ match,
    booth, robot/shop name or question word: passers-by talking). **only with a customer**: listens only while the
    vision module (2.3, `PRESENCE=1 ./run_vision.sh`) reports someone in front, and 8 s after each answer.
  - The list shows what was heard, the confidence, the recognition time, and what was done or why it was ignored.
    Audio is never stored (PDPA); only the last 20 texts are kept, in memory.
  - Mic: `--listen-source laptop` (default; `--listen-device <PulseAudio source>` to pick one, else the default input
    or, if it is silent, the next live one) or `robot` (G1 mic multicast 239.168.123.161:5555, needs the robot's
    voice assistant in wake-up mode; untested).
- **Caption**: the phrase being spoken, highlighted in the whole text.
- **Point to a booth**: one button per booth of the active shop. Each shows the arm (L/R), angle and distance;
  booths the robot cannot point at are greyed, with the reason as tooltip.
  - The arms are taken over automatically.
  - The robot turns its torso a little and raises the arm with an open palm (low for near booths, nearly
    horizontal for far ones).
  - It holds for the set time (or until the next action, with the checkbox) and returns to rest.
  - **Rest** brings the arms back to rest at any time. The current action and its phase are shown under the
    buttons, and the 3D view draws the line to the booth.
  - The speed slider also sets how fast the arm moves: at 0.5 a raise takes about 3 s, at 1.0 about 1.5–2 s.
  - The booth sentence ("คาเฟ่อยู่ทางนี้ค่ะ" / "Cafe is this way, please.") starts as the arm arrives.
- take over / release the arms, speed scale, test poses (S0);
- hand poses;
- balance indicator;
- joint table;
- **3D view**: the robot's measured pose (sim or real) among the active shop's booths, rendered by the server.
  Camera behind the robot, facing it, or from above, with an optional line from the shoulder to a booth.

**Scripts & voice** tab (`/#scripts`): every text the robot speaks, Thai and English (robot and shop name,
introduction, service, pointing sentence, filler, "no information"), with gesture tags `[welcome]`, `[present]`,
`[both]`; voice choice, test sentence, **upload a reference voice** (5–12 s + transcript + consent; Thai is then
spoken in that voice, English stays on the default voice); **motion learned from video** (the wai clip with its
video-vs-robot comparison and an on/off switch, the keep-alive style).

**Shop** tab (`/#shop` opens it directly): the map editor.
- **Upload** a background map (PNG/JPEG/WebP, up to 30 MB).
- **Scale** tool: click two points and type their real distance. **Origin** tool: click the point that becomes
  (0, 0). By default, robot and booths stay on the map when you rescale or move the origin.
- **Robot**: click to place it, drag to set its heading. **+ Booth**: drag a rectangle.
- **Select**: drag a booth, its aim point (red dot), its corner handle or the robot. The side panel edits names
  (TH/EN), size, rotation, aim point and height, colour, descriptions and labels.
- For the selected booth, the panel shows what the robot sees: side, angle, distance, torso turn, and which arm
  it uses. Booths behind the robot are greyed.
- **Save** writes `shops/<name>/shop.json` (the previous version is kept as `shop.prev.json`). **Use for robot**
  makes the shop active (pointing, 3D view).
- **New** and **Copy** create shops. Keyboard: Delete removes the selected booth, Esc returns to Select,
  Ctrl+S saves.

## How the robot is driven (same as on the real robot)

- **Legs**: Unitree's own controller (standing). exp05 never commands them.
- **Waist yaw + both arms**: `rt/arm_sdk` (unitree_hg LowCmd_, motors 12–28; `motor_cmd[29].q` = weight 0..1).
  `deploy/arm_channel.py` is the only code that writes it. Its rules:
  - Take-over: the weight ramps up over 1.5 s with the targets at the measured pose.
  - Release: the weight ramps down over 1.5 s.
  - **E-STOP = release** (never damping).
  - Every target is kept inside the joint limits with a margin, and its speed is limited to 2 rad/s × the speed
    scale.
  - The arms are released automatically if the motion layer stops sending goals or the robot state goes stale
    (0.5 s).
- **Motion** (`deploy/motion.py`): actions are clips sampled at 50 Hz. Each joint follows minimum-jerk segments,
  sized so the peak speed stays under 80% of the channel's limit. Hand goals switch at set times. A new action
  starts from the current goal, so it never jumps.
- **Pointing** (`deploy/pointing.py`):
  - Booth → angle and distance. The arm on that side points (the right one straight ahead); the torso turns
    0.25 × the angle, at most 15°.
  - Elevation from the distance (−40° at 1.5 m … −3° at 40 m).
  - IK puts the wrist on the ray, the fingers along it, and the palm turned up. The elbow stays slightly bent.
  - Plans are cached per shop. Details are in PLAN.md 3.4 "As built".
- **Hands**: the motion layer uses Inspire values (12, right 0–5 / left 6–11; pinky, ring, middle, index, thumb
  bend, thumb rotation; 1 = open). `--hands inspire` sends them on `rt/inspire/cmd` (sim and RH56);
  `--hands dex3` maps them onto Dex3-1 (`rt/dex3/{left,right}/cmd`, 7 motors per hand) for the robot as it is now
  (the default; the sim answers on the same topics). With Dex3, "flat" and "relaxed" fold the thumb across the
  palm: an open Dex3 thumb sticks out 8.8 cm and would hit the other thumb in the wai and the hips at rest.
  Even folded, the Dex3 thumb base stands ~3.2 cm proud of the finger pads (any thumb angle), so Dex3 palms cannot
  meet flat: the wai turns each hand 35° about the vertical (`gestures.WAI_HAND_TURN_DEG`; Inspire 0), the finger
  edges close in front (~3.7 cm), the thumb bases move back toward the chest (user's choice 2026-11-13).
- **Behaviour** (`deploy/behaviour.py`, `gestures.py`, `scripts.py`): each action is one timeline (motion clip +
  speech items); the server adds the idle layer (`gestures.idle`: slow sines, breathing, a rare 3° waist glance,
  finger drift) on top of every clip and while standing, so the robot never freezes.
- **Speech** (`deploy/speech.py`): TTS through the AI service, cached in `cache/tts/` (fixed texts are pre-rendered
  when a shop becomes active); audio starts on the clip's clock minus the output's latency; captions are the TTS
  phrases. **Q&A** (`deploy/qa.py`): language from the letters, FAQ similarity ≥ 0.72, else the LLM with the shop
  context (booths with "where" seen from the entrance, FAQ, service text); booth from the answer or the question
  (names and `labels` synonyms).
- Waist roll/pitch are locked on our robot. They are not in the model, and their slots report 0.

## Simulation (`g1/deploy/sim_robot.py`)

- **Model**: `g1/model/g1_dex3.xml` (Dex3-1, from unitree_rl_gym's `g1_29dof_with_hand_rev_1_0.urdf`) or
  `g1/model/g1_inspire.xml` (from unitree_ros' `g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf`), built by
  `g1/tools/build_model.py --hands dex3|inspire`:
  - the waist roll/pitch joints are made fixed;
  - Unitree motor armature/friction on the body joints;
  - PD actuators set per command, like the real motor drivers;
  - hands: Dex3 7 motors per hand on `rt/dex3/<side>/cmd|state` (as on the robot); Inspire 6 actuated DoF per
    hand on `rt/inspire/cmd|state` (the mimic joints are equalities);
  - sites for the palms (same pose on the wrist for both hands, so the IK gives the same arm angles), shoulders
    and head;
  - 34.4 kg (Dex3) / 33.4 kg (Inspire).
- **Unitree standing**: the pelvis is pinned at standing height; legs and arms hold a standing posture by PD.
  - `rt/arm_sdk` blends in by its weight, as on the robot.
  - When the arm_sdk stream stops, the arms fade back over 1 s.
- **Balance indicator** (`rt/sim/balance`, shown on the page): the whole-body COM and the ZMP, and their margin
  inside the feet's support polygon. At rest the margin is 8.4 cm; with an arm raised forward it is 5.3–6.6 cm; while pointing ≥ 5.7 cm.
  An instant (unramped) take-over drives it to −42 cm, which is why the arm channel always ramps.
  - Why not a free-standing sim: we don't have Unitree's balance controller, and plain leg PD made the robot fall
    at once (a 33 kg robot needs about 230 N·m/rad of ankle stiffness).
- `--shop shops/<name>/shop.json` puts the booths around the robot as translucent boxes with their aim points
  (robot frame: the robot is at the origin, facing +x).

## Student pack (True Alpha Co-Build Phase 2, 2026-11-13)

Students run exp05 in simulation on their own laptops. Sources are in `student/`:
- `STUDENT_GUIDE.html`: agenda, one section per session 1.1–2.3, integration, reference, troubleshooting;
- `README.md` and `install.sh`;
- `img/`: screenshots.

Build the zip with:

```bash
tools/make_student_pack.sh        # -> dist/exp05_student_pack.zip (~28 MB)
```

The zip holds only code and small data. It does not include the venvs, the models, the YouTube videos, or the
comparison sheets that contain video frames. `install.sh` downloads the large files:
- uv and the `g1/`, `vision/` and `ai/` environments;
- the vision models;
- TTS, ASR and the LLM from Hugging Face.

`--no-llm` writes `ai/.no_llm`, which `ai/run_ai.sh` honours, and `--no-ai` skips the AI part. Disk: about 2 / 11 / 19 GB.
The installer runs no `point_check`, because the first pointing plans can take more than its 60 s timeout on a busy laptop.
Tested 2026-11-13 from a clean unzip: `--no-llm` and `--no-ai` pass, and `vision_check` passes 26/26 in the copy.
**Rebuild the zip after any change to code, clips or the guide.** The guide inlines `student/img/*.jpg` at build
time, and its relative link `ASR_LIBRARIES.md` is copied next to it.

## Layout

```
exp05/
├── PLAN.md, README.md, TESTING_ROBOT.md
├── run_sim.sh, run_robot.sh   # sim + AI service + server / real robot + AI service + server
├── ai/                        # uv project (torch): service.py (:8105 TTS + LLM + ASR), clone_engine.py (F5-TTS-THAI),
│                              #   run_ai.sh, tools/video_to_wai.py, tools/video_to_idle.py, models/ (HF cache), voices/
├── clips/                     # wai_dex3.json / wai_inspire.json (learned wai), idle.json (keep-alive style), source/ (measurements, sheets)
├── videos/                    # examples/ (downloaded test videos), SOURCES.md
├── cache/                     # tts/ (rendered speech), poses_<hands>.json (gesture IK)
├── shops/<name>/              # shop.json (robot pose, booths, FAQ), background map image
│   └── mockup/                # 36 x 30 m mockup shop + make_mockup.py + preview.png
├── tools/storage.sh           # disk use vs the 200 GB limit
├── tools/make_student_pack.sh  # student zip -> dist/exp05_student_pack.zip
├── vision/                    # 2.3 people detection (own uv env, port 8107): run_vision.sh, vision/*.py, tools/vision_check.py, models/ (downloaded)
├── student/                   # student pack sources: STUDENT_GUIDE.html, README.md, install.sh, ASR_LIBRARIES.md, img/
├── dist/                      # built student zip
├── logs/                      # per run (run_sim.sh)
└── g1/                        # uv project (Python 3.10): .venv, model/, deploy/, tools/, third_party/
    ├── deploy/                # sim_robot.py, server.py (:8100), arm_channel.py, robot_model.py, shops.py, view3d.py,
    │                          #   pointing.py, motion.py, gestures.py, behaviour.py, scripts.py, speech.py, qa.py, video_jobs.py, listen.py,
    │                          #   static/index.html, static/monitor.html
    ├── model/g1_dex3.xml, g1_inspire.xml   # generated
    ├── tools/                 # build_model.py, render_check.py, sim_check.py, server_check.py, point_check.py,
    │                          #   behaviour_check.py, video_check.py, record_check.py, listen_check.py, make_wai_clip.py, point_render.py, ui_shot.py (--point BOOTH)
    └── third_party/           # unitree_sdk2_python (copy from exp04), unitree_ros G1 + Inspire URDF/meshes,
                               #   unitree_g1_dex3 (G1 + Dex3-1 URDF/meshes from unitree_rl_gym), silero_vad (VAD model, MIT)
```

Ports and domains: UI 8100 (tests 8101/8102/8104/8109), AI service 8105, DDS domain 5 in the sim (tests 51–54, 56). exp05 runs
next to exp02/exp04 (it uses its own ports and domain).
