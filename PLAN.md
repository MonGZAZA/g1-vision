# exp05: G1 shop greeter (plan, updated 2026-11-09, not implemented yet)

The G1 stands at the shop entrance, facing into the shop, and stays there. It interacts with the upper body only
(arms, 5-finger hands, a small waist turn) and a female synthesized voice (Thai or English). Three separate tasks,
each started by a button:

1. **Introduce itself**: Thai *sawasdee* greeting (wai: the hands are raised, no bow), learned from a video, then
   gentle "keep-alive" arm/hand motion while it speaks the introduction. The wai is done in both languages.
2. **Introduce the service and answer typed questions**: speaks a script or an answer from an offline LLM, with
   small talking gestures.
3. **Point to a booth**: an open palm shows the way to a selected booth. The direction always aims at the booth.
   A near booth gets a lower arm and a far booth a more horizontal arm. The torso turns a little toward the booth,
   just enough to be noticed, and the arm does the rest.

Status: plan + mockup shop (`shops/mockup/`). The real wai and keep-alive videos and the real shop map will come
from the user later.

## Decisions (user, 2026-11-09)

| Topic | Decision |
|---|---|
| Head/torso toward the target | **Small torso turn** only (enough to notice the direction), the arm covers the rest |
| Wai | Raise the hands only: **no bow, no knee dip** |
| Wai with Dex3 (2026-11-13) | Dex3 palms cannot meet flat (thumb bases ~3 cm proud): **hands turned 35° inward**, fingertips close in front. Inspire: palm to palm |
| Hands | **5-finger hands** = **Inspire RH56** (the only 5-finger option in exp02's hand list: none / Dex3-1 / Inspire RH56). Our G1 currently has Dex3-1, so the code supports both, Inspire is the default |
| Voice | **Female** ("kha"). **Thai and English**; the wai is always done, in both languages. **Start with a TTS model's built-in voice**; the operator can upload a reference voice later (voice cloning) |
| Sound | **Robot speaker**; the laptop speaker while testing |
| Q&A | **Offline LLM**; questions are **typed** for now |
| Captions | **Testing only** (operator UI). At deployment: no customer-facing caption; optional caption for monitoring from a remote machine |
| Pointing | **Open palm** |
| Platform | Runs on **this laptop** (RTX 4060, 8 GB, shared with the other experiments). **Simulation first**; the real robot is connected later (S7) |
| Shop | Up to **40 × 40 m**. Background map uploaded in the UI; robot and booth positions saved. Mockup shop until the real one comes |

---

## 1. Constraints from our robot (G1 EDU, see exp02/exp04)

| Fact | Consequence for exp05 |
|---|---|
| Waist roll + pitch are locked (`g1_29dof_lock_waist`), waist yaw works | No bow anyway (the user does not want one). The small torso turn uses waist yaw |
| The G1 has no neck joint, the head is fixed to the torso | "Head toward the target" = the small waist-yaw turn (3.4) |
| Our own walking/standing policy is not ready (locked-waist retrain still open) | **Lower body = Unitree's own controller** in normal mode, standing (as in exp04). We never take the legs |
| `rt/arm_sdk`: upper-body override while Unitree's controller balances (waist + both arms, weight in slot 29) | All exp05 arm motion goes through `arm_sdk` at 50–100 Hz. Ramp the weight 0 → 1 to take over and 1 → 0 to hand back. No debug mode, no `rt/lowcmd` |
| **5-finger hands**: our G1 had Dex3-1 (3 fingers) in exp02. The plan assumes **Inspire RH56** (5 fingers, 6 DoF: pinky, ring, middle, index, thumb bend, thumb rotation) on `rt/inspire/cmd|state`, already supported by `hands.py` (exp02/exp04) | Finger poses: flat hand with fingers together (wai, open-palm guide), relaxed, slightly curled. The sim needs a G1 + Inspire model (Unitree publishes one with its teleoperation repos; exp04 has none yet). **To confirm: which 5-finger hand is fitted** |
| Audio: `G1AudioClient.PlayStream` (16 kHz mono PCM to the robot speaker), `SetVolume`, `LedControl`; its built-in `TtsMaker` has no Thai | TTS runs on the laptop; audio goes to the robot speaker (deployment) or the laptop speaker (testing). The head LED can show "speaking" |

**Never send damping while the robot stands under Unitree's controller.** exp05's E-STOP = stop speech, ramp the
`arm_sdk` weight to 0 within ~1 s (arms go back to Unitree's posture). The real emergency stop stays on the Unitree remote.

---

## 2. Architecture

```
 Operator UI (browser, :8100) ── also openable from a remote machine for monitoring (caption, status)
        │  HTTP / server-sent events
        ▼
 ┌──────────────── exp05 server (Python, laptop) ───────────────────┐
 │ shops/<name>/shop.json + background image (robot pose, booths,  │
 │   descriptions, FAQ, scripts)                                    │
 │ Behaviour: Intro / Service / Answer / Point → Timeline           │
 │ Timeline player (one clock): motion + audio + caption events     │
 │ Motion: clip library (wai, gestures) + procedural idle + pointing IK
 │         + blender (min-jerk transitions, joint/velocity limits,  │
 │           self-collision check, reach envelope)                  │
 │ TTS (female, Thai + English, cached per phrase) · offline LLM    │
 └──────┬───────────────────────────────┬───────────────────────────┘
        │ rt/arm_sdk + rt/inspire (DDS)  │ audio: PlayStream (robot) or laptop speaker (test)
        ▼                                ▼
   Real G1 (Unitree standing)  or  MuJoCo sim (G1 lock-waist + 5-finger hands, same DDS topics)
```

- New, self-contained uv project `exp05/g1/` (Python 3.10, like exp04). It copies only what it needs from exp04:
  the Unitree SDK, MJCF models, `hands.py`, the posture view, and the sim-robot skeleton. There is no walking
  policy, lidar or navigation, so it can run next to exp02/exp04. Heavy offline tools (video pose estimation,
  retargeting, TTS, LLM) get their own environments under `exp05/tools/`; files are the interface.
- Ports: UI 8100, tests 8101–8104. DDS domain 5 in the sim. Real robot: same wired adapter/profile as exp02
  (`enx00e04c80fe28`, `g1-wired`).
- **Sim** (`run_sim.sh`): MuJoCo G1 `g1_29dof_lock_waist` + 5-finger hands. It publishes `rt/lowstate` and the
  hand state, and it obeys `rt/arm_sdk` and the hand commands like the real robot. Two modes:
  - **A, pinned pelvis** (default): for designing and previewing motion, pointing accuracy and timing.
  - **B, free-standing**: legs held by a standing controller (exp04's `unitree_v0` at zero command, or stiff PD).
    This shows whether the arm motion pushes the robot. It is only an indication: Unitree's real balance
    controller is different.

---

## 3. Motion design

All motions are joint-space trajectories for the 15 `arm_sdk` joints (waist yaw + 2 × 7 arm joints) plus the
finger joints (2 × 6). They are stored as clips (`.npz`: time, q, finger q, phase marks, metadata) at 50 Hz.
Every motion passes the same checks before it can play:
- joint limits with a margin, and velocity/acceleration limits;
- self-collision in MuJoCo (hand–hand, hand–torso, hand–thigh, with a minimum distance);
- a reach envelope: hands stay inside a box in front of the robot, so they don't swing far toward people.

### 3.1 Rest pose
Polite shop-staff stance: arms relaxed, hands loosely in front of the lower belly or beside the thighs (chosen
from the user's video). Waist yaw 0 (facing into the shop). Every action starts and ends here (min-jerk
transition, 0.8–1.5 s).

### 3.2 Sawasdee / wai (task 1), learned from video, hands only

Pipeline "video → robot clip", built once on example videos and re-run when the user's video arrives:

1. **Body motion from monocular video**: GVHMR (world-grounded SMPL-X). Fallback: WHAM / 4D-Humans.
2. **Hands**: HaMeR (MANO hand pose per frame) for finger shape and palm orientation.
3. **Retarget to the G1**: GMR (General Motion Retargeting, supports `unitree_g1`, reads GVHMR output). Keep arms
   only. Waist yaw = 0 during the wai, and legs/pelvis are discarded (Unitree holds them). Any bow or knee bend
   in the video is ignored.
4. **Wai-specific correction**: a small IK step makes the palms meet as mirror images in front of the chest. Keep
   a 1–2 cm gap (no pressing, protects the fingers), fingertips up. The hand height is a parameter (default:
   thumbs at chin level, the polite level for guests).
5. **Fingers**: 5-finger flat pose (fingers straight and together, thumb along the index), mapped from the HaMeR
   finger angles to the 6 hand DoF.
6. **Clean-up**: low-pass filter (~4 Hz), time-warp to the target length, symmetry check, the checks above.
   Phase marks: `raise` → `hold` → `lower`. The timeline puts "สวัสดีค่ะ" / "Hello" in the `hold`.

"Learn from the video" means **imitation by retargeting**: the robot copies the human's arm and hand motion
kinematically. No RL tracking training is needed, because the legs are not involved and Unitree's controller
balances. If the arm motion disturbs balance (sim mode B or the robot), the fix is slower playback.

### 3.3 Keep-alive (after the wai, while introducing itself) and talking gestures (task 2)

Two layers on top of the rest/base pose:

- **Idle layer (always on, low amplitude)**:
  - smooth noise (sums of slow sines, 0.1–0.4 Hz) on shoulder, elbow and wrist, ~2–4° amplitude;
  - a "breathing" component (~0.25 Hz);
  - small finger drift;
  - an occasional slow waist-yaw glance (±3–5°).

  The robot is never frozen, and the amplitude and speed are set from the user's keep-alive video.
- **Gesture layer (while speaking)**: short clips such as an open-palm "presenting", two-hands "explaining", small
  beat gestures, and a "welcome" sweep toward the shop.
  - Sources: the user's keep-alive video (same video pipeline), and until then the **BEAT2** co-speech gesture
    mocap (SMPL-X, retargeted with GMR; CC BY-NC, research use).
  - They start at phrase starts (from the TTS timing). Optional tags in the script (e.g. `[present]`,
    `[welcome]`) force a gesture.
  - Amplitude limits are stricter than for the wai, because customers are close.

### 3.4 Pointing to a booth (task 3), procedural, open palm

Inputs: the robot pose in the shop (x, y, yaw from the UI) and the booth's **aim point** (x, y, z; default = booth
centre at display height 1.0 m, editable per booth).

1. Aim point in the robot frame → azimuth θ (+ = left) and horizontal distance d.
2. **Arm choice**: left arm for targets on the left (θ > +5°), right arm on the right, either arm straight ahead
   (configurable default). The pointing arm never crosses the body.
3. **Small torso turn** (head turns with it): ψ = clamp(k·θ, ±ψmax), with **k = 0.25 and ψmax = 15°** as defaults
   (UI sliders). This is a visible cue, not facing the booth. The arm covers θ − ψ, up to ~75° outward for a booth
   at the side.
4. **Elevation follows distance**: the arm aims along the ray from the shoulder to the aim point, plus a tunable
   curve `elev(d)` so the near/far difference is easy to read. Example: −40° at 1.5 m, −25° at 3 m, −10° at 10 m,
   −3° at 40 m. Near → clearly lower, far → nearly horizontal. The range covers d = 1–45 m (a 40 × 40 m shop
   seen from its entrance).
5. **Hand**: open palm, fingers together, palm turned slightly up and toward the booth, wrist in line with the
   forearm.
6. **IK**: damped least-squares in MuJoCo on the 7-DoF arm (waist fixed at ψ). Goal: forearm and palm along the
   ray, elbow slightly bent, natural elbow swivel. Joint-limit and self-collision aware.
7. **Timeline**: torso starts to turn → arm raise (min-jerk, ~1 s) → hold while speaking ("ร้านกาแฟอยู่ทางนี้ค่ะ" /
   "The café is this way") with idle on top → optional small "emphasis" push along the ray → return to rest. The
   other arm stays at rest with idle.

Checks in the sim:
- the mockup shop's 12 booths, plus a grid of virtual booths (θ −100…+100°, d 1–45 m);
- azimuth error of the pointing ray (shoulder → palm) **≤ 5°**;
- near booths always lower than far ones;
- no limit or collision violations.

**As built (S2, 2026-11-10)**: `g1/deploy/pointing.py`, `motion.py`; checks in `g1/tools/point_check.py`.
- Arm: left above +5°, right below −5°, right in between. The pointing ray always stays at least 3° outward
  of the torso's facing. For a near booth almost straight ahead (it lies slightly inward of the right shoulder), the
  torso turns a few degrees more instead (up to the 15° limit).
- Elevation curve: −45° at 1 m, −40° at 1.5 m, −25° at 3 m, −10° at 10 m, −3° at 40 m (log-distance
  interpolation), plus the aim point's height above 1 m. The 3D view's red line goes to the aim point itself, so
  for near booths the arm is lower than that line on purpose.
- IK in two stages from several seeds: the wrist onto the ray first, then the palm orientation blends in.
  - The palm is turned 50° up from facing inward.
  - The elbow is kept between 0.5 and 1.35 rad and pulled toward 1.0. On this model a positive elbow angle
    straightens the arm (straight ≈ 1.45); beyond that the joint bends backwards, which is allowed but unnatural.
  - The upper arm never swings in toward the torso.
  - The posture with the fewest joints near a limit wins.
  - About 0.15–0.25 s per booth, cached per active shop.
- Motion: the torso leads by 0.25 s, then a min-jerk arm raise with the hand opening flat. Durations follow the
  joint distance, so the peak speed stays under 80% of the arm channel's limit (about 1.5–2 s at speed 1.0).
  During the hold the arm sways slightly (±1.5°). Then the arm returns and the hand relaxes. Any new action starts
  from the current pose.
- Results:
  - 12 mockup booths + 168 grid targets: worst azimuth error 3.5°; no collision or limit violation; near always
    lower than far.
  - Live in the sim: the ray is within 0.5° of the target. It sits 1–2° lower than planned, because the PD
    position control sags under gravity.
  - ZMP margin stays ≥ 5.7 cm.
- The rest pose is now Unitree's: shoulder pitch 0.3, elbow 0.9 (S0 had the forearms raised by mistake).

---

## 4. Voice, captions and sync

- **Female voice, Thai and English, offline on the laptop.** Candidates to compare by listening:
  - **Thai**:
    - **ThonburianTTS** (Mahidol, F5-TTS based, best reported intelligibility);
    - **F5-TTS-THAI** (VIZINTZOR);
    - **MMS-TTS-THAI female** (VITS, light, CPU), as the fallback.
  - **English**:
    - **F5-TTS base**, cloning the same female reference voice as the Thai model, so both languages sound like one
      robot;
    - or a light English voice (e.g. Kokoro).

  First version: a built-in/default female voice of the chosen models (no recording needed). The F5 models can clone
  a voice from a 5–10 s reference recording: the UI gets **"Upload voice"** (WAV/MP3 + its transcript, consent
  checkbox), then re-renders the cached fixed texts with the new voice. One active voice per shop, kept with it.
  Thai text with English product names needs pronunciation overrides (spelled-out forms in the script).
- **Language per action**: a TH/EN switch in the UI (default Thai). A typed question gets its answer in the
  question's language. The wai always comes first in the introduction, in both languages.
- **Phrase-level synthesis + cache**: texts are split into phrases (PyThaiNLP for Thai, punctuation for English) and
  synthesized one phrase at a time, so each phrase's exact audio length is known. Fixed texts (introductions,
  booth sentences) are pre-rendered when saved, so the buttons start instantly. Answers are synthesized on the
  fly, and the first phrase plays while the rest is rendered.
- **Audio output**: robot speaker via `PlayStream` (resampled to 16 kHz mono) for deployment; the laptop speaker in
  testing and in the sim. A switch in the UI chooses. Each output's latency is measured once and compensated.
- **Timeline**: every action compiles to one timeline with three tracks: motion (clips/procedural segments with
  phase marks), audio (phrases) and caption events. One clock plays all three. Target: the voice within ±100 ms
  of the motion marks (e.g. "สวัสดีค่ะ" lands in the wai `hold`).
- **Captions** are the timeline's caption events. During testing they show in the operator UI in time with the
  voice. At deployment there is no customer screen; the same caption and status can be watched from a remote
  machine by opening the operator UI over the network (read-only "monitor" view).

## 5. Q&A (task 2), offline

- **Input**: the operator types the question (Thai or English), or picks a frequent one from the FAQ. No
  microphone in this version.
- **Answer**: the shop knowledge in `shop.json` (service text, booth names and descriptions, FAQ) is searched
  first. An exact FAQ match is answered directly. Otherwise a **local LLM** phrases a short answer from that
  knowledge only (≤ 2–3 sentences, polite female particles "ค่ะ/นะคะ", same language as the question).
  - Candidates: a Thai-capable small model run with llama.cpp/Ollama at 4-bit, e.g. Typhoon 2.x (SCB 10X, Thai)
    or Qwen3 4B/8B.
  - Budget: the 8 GB laptop GPU is shared with TTS, so the target is ≤ 4–5 GB for the LLM, with a CPU fallback.
- The operator sees the answer and can edit it before "Speak" (an "auto-speak" switch for later).
- **Filler while thinking**: a short phrase ("ขอสักครู่นะคะ" / "One moment, please") + a "thinking" gesture.
- **Hand-off to pointing**: if the answer names a booth, the UI shows "Point to <booth>" next to Speak (one
  click). It can be chained after the answer automatically (switch).

---

## 6. Interface (operator UI, :8100)

| Panel | Contents |
|---|---|
| **Shop map** (top view, editable) | **Upload a background map** (PNG/JPG floor plan or an exp04 lidar map). Set its scale: click two points and type the real distance, or type m/px. Set the origin. **Robot**: drag to place, rotate for its heading. **Booths**: add, drag, resize (rectangle, optional rotation), Thai/English name, aim point (x, y, z), descriptions, labels. The small-turn sector and the pointing ray of the selected booth are drawn. A warning appears for booths outside the pointing range (behind the robot). **Save** writes `shops/<name>/shop.json` + the image; **Load**/**New** switch shops |
| **3D preview** | Posture view of the robot (sim or real state) with booth markers and the pointing ray. "Preview in sim" runs any action on the simulated robot before the real one |
| **Actions** | Big buttons: **Sawasdee + introduce**, **Introduce service**, **Answer** (question box → answer → Speak), **one button per booth** "Point to <name>", **Rest**, **Stop** (stop speech, return to rest), **E-STOP** (stop speech, release `arm_sdk`). TH/EN switch. The queue shows what is playing |
| **Voice** | Built-in voice choice, **upload a reference voice** (clone), test sentence, re-render all fixed texts |
| **Scripts** | Thai and English texts for the self-introduction and the service introduction, optional gesture tags, per-booth sentence template ("<name> อยู่ทางนี้ค่ะ" / "<name> is this way"), FAQ list, pronunciation overrides. Voice, speed, volume. "Render" pre-builds audio + timing |
| **Caption / monitor** | Live caption in time with the voice (testing), and a read-only monitor view for a remote machine (caption, current action, robot status) |
| **Robot / safety** | Sim/real, link status, Unitree mode (must be standing), `arm_sdk` weight, hand status, joint-limit/temperature warnings, speed scale (0.3 … 1.0) for first tests, audio output (robot/laptop), torso-turn k and ψmax, elevation curve |
| **Motion library** | Clips (wai, gestures) with the source video next to the robot playback, to judge the retargeting |

### Shop file format (`shops/<name>/shop.json`, made by the mockup generator and by the UI)

- **Frame**: metres and degrees. x to the right, y into the shop, origin = inside south-west corner. Yaw 0 = +x,
  90 = +y (counter-clockwise).
- `background`: `image`, `resolution_m_per_px`, `size_px`, `origin_bottom_left_m` (world position of the image's
  bottom-left corner).
- `robot`: `x`, `y`, `yaw_deg`.
- `booths[]`: `id`, `name_th`, `name_en`, `rect {x, y, w, h, angle_deg}`, `aim {x, y, z}`, `description_th/en`,
  `labels`, `color`.
- `faq[]`: `q_th/q_en`, `a_th/a_en`, optional `booth` (for "point to").
- Scripts are added to the same file by the UI.

### Mockup shop (`shops/mockup/`, made 2026-11-09)

- **Shop**: 36 × 30 m electronics store. The entrance is in the middle of the south wall. The robot is at
  (18.0, 1.2), facing into the shop (yaw 90°).
- **Booths (12)**, chosen to cover the pointing cases. Values are as seen by the robot: azimuth θ, distance d, and
  the torso turn ψ with the defaults above.

  | Booth | θ | d | ψ | Case |
  |---|---|---|---|---|
  | New Arrivals table | +37° | 3.2 m | +9° | near |
  | Information | −67° | 6.0 m | −15° | |
  | Cashier | +72° | 7.4 m | +15° | |
  | Cafe | −77° | 14.4 m | −15° | |
  | Restrooms | +86° | 16.0 m | +15° | almost to the side |
  | Phones & Tablets | +50° | 13.7 m | | |
  | Computers | −50° | 13.7 m | | |
  | Robot Showroom | 0° | 15.3 m | 0° | straight ahead |
  | Smart Home | +31° | 24.3 m | | |
  | Audio | −33° | 23.7 m | | |
  | Gaming Zone | +13° | 26.7 m | | far |
  | Service Center | −18° | 27.4 m | | far |

- **Files**:
  - `map.png`: the background plan, 950 × 825 px at 0.04 m/px. It is what a user would upload: walls, shelves,
    booth areas with names, scale bar. The robot is not drawn.
  - `shop.json`: the layout and a 3-entry mock FAQ.
  - `preview.png`: robot + pointing rays + the table above.
  - `make_mockup.py` regenerates all three.
- The real shop replaces this through the UI upload; the mockup stays as a test fixture.

---

## 7. Example videos (until the real ones come)

These are used to test the video → GVHMR/HaMeR → GMR pipeline end to end and to choose the gesture style.
Download them to `exp05/videos/examples/` (untrusted data, own folder). Record the source URL and license in
`videos/SOURCES.md`. They are not for redistribution.

| Need | Example source | Notes |
|---|---|---|
| Wai / sawasdee, front view, upper body | Thaiger "How to Wai like a Thai, with Som"; Thailandblog's wai explainer video; YouTube "HOW TO WAI LIKE A THAI PERSON" (Real Thailander) | Pick a standing person, both arms and hands visible, steady camera. Skip The Thailand Life's video (its own note says the presenter's wai is wrong) |
| Shop-staff wai + welcome (closest to the real use) | Thai shop/hotel staff greeting clips (search "พนักงานไหว้ต้อนรับ", "สวัสดีค่ะ ยินดีต้อนรับ") | To choose after viewing; the user's video replaces it |
| Keep-alive / talking gestures | **BEAT2** dataset (mocap, SMPL-X, speech + gestures; CC BY-NC) via PantoMatrix, plus 1–2 front-view videos of a presenter talking calmly | Filter for calm/neutral clips |
| Open-palm guiding (reference only; pointing is procedural) | Hotel/airport staff "this way please" clips | Sets the hand shape and timing only |

**Specs for the user's real videos** (for the people who record them):
- **Setup**: one person, front view, camera at chest height, 2–3 m away, on a tripod. The whole upper body and both
  hands in the frame. 30–60 fps, 1080p, plain background, short or tight sleeves.
- **Takes**:
  1. Wai 3–5 times at normal speed, saying "สวัสดีค่ะ" (and once "Hello, welcome"), hands only, no bow.
  2. 1–2 min standing and talking calmly about the shop (keep-alive and talking gestures).
  3. Optional: open-palm guiding to near and far places.
- One side-view take of the wai helps with palm and elbow depth.

---

## 8. Stages

| Stage | Work | Done when |
|---|---|---|
| S0 ✅ | `exp05/g1` uv project; MuJoCo lock-waist model with 5-finger hands; sim robot with `arm_sdk` + hand topics (modes A/B); server skeleton on :8100 | Sim arms follow `arm_sdk` with the weight ramp; fingers follow commands |
| S1 ✅ | Shop editor: upload background, scale/origin, robot pose, booths, save/load (mockup as the first shop); 3D preview | Mockup loaded, edited, saved and reloaded identically; a new image can be uploaded and scaled |
| S2 ✅ | Pointing: arm choice, small torso turn, elevation curve, IK, open palm, checks, one button per booth | All mockup booths + grid: azimuth error ≤ 5°, near lower than far, no limit/collision violations |
| S3 ✅ | TTS comparison (Thai + English female), voice choice, cache, phrase timing, timeline player, captions, laptop/robot audio switch | Booth sentence + pointing play in sync in both languages; user picks the voice |
| S4 ✅ | Video pipeline (GVHMR + HaMeR + GMR + wai correction) on example videos → wai clip | Wai clip passes the checks and looks right next to the source video (user judges) |
| S5 ✅ | Keep-alive idle + gesture library (BEAT2 / video) + scheduler; full "introduce itself" (TH/EN) and "introduce service" | Both actions run end to end in the sim; the robot is never frozen; user judges the motion |
| S6 ✅ | Offline Q&A: typed question → FAQ / local LLM answer → preview → speak with gestures → "point to" hand-off | Correct answers from the mockup shop data in Thai and English, within the GPU budget |
| S7 (prepared) | Real robot, step by step (a written test procedure like exp02's). Robot standing in Unitree mode, a person at the remote. Steps: `arm_sdk` take-over/release at speed 0.3; each motion slow, then at full speed; torso-turn range; hands; robot speaker + latency calibration; remote monitor view | Every action on the real G1 at full speed, no balance problems, E-STOP release checked |
| S8 | User's videos replace the examples (re-run S4/S5); real shop map uploaded, booths placed; dry run at the shop entrance | User approves at the shop |

**As built (S3–S6, 2026-11-10)** — what changed from the plan above:
- **Video pipeline: MediaPipe Holistic instead of GVHMR + HaMeR + GMR.** No SMPL-X/MANO registration is needed,
  and the example videos are chest-up close-ups where the elbows are out of the picture, so a joint-by-joint
  retarget would invent the arm. The video gives what it really shows: the raise/lower timing curve, the hold
  height (fingertips relative to chin/nose), the forward tilt of the hands and the finger straightness. The robot's
  arm comes from IK (`gestures.solve_wai`: palms mirrored with a 1.5 cm gap, path through a "hands 8 cm apart" pose
  so the palms only close at the end). `ai/tools/video_to_wai.py` → `g1/tools/make_wai_clip.py` → `clips/wai.json`
  (+ comparison sheet). With a full upper-body video (the specs above) the same measurements are more reliable.
- **Keep-alive**: a procedural idle layer (sums of slow sines, breathing, rare waist glance, finger drift) on top of
  every action and while standing; gestures (welcome, present, both, beat) from IK poses, scheduled at TTS phrase
  starts and `[tags]` in the scripts. BEAT2 was not used (SMPL-X). `ai/tools/video_to_idle.py` scales the idle and
  the beat rate from a talking video (`--install`); the downloaded presenter videos were too lively for a greeter.
- **TTS**: Thai = VIZINTZOR/MMS-TTS-THAI-FEMALEV2 (VITS, fast), English = Kokoro-82M `af_heart`; uploaded voice =
  F5-TTS-THAI cloning for Thai (English stays Kokoro: the clone model is Thai-only). Phrase-level synthesis gives
  exact phrase times; captions start within ±60 ms of the speech in the sim.
- **LLM**: scb10x/typhoon2.5-qwen3-4b, 4-bit (bitsandbytes), 2.9 GB GPU, ~1 s per answer, thinking off; answers
  only from the shop context; wrong-language answers fall back to the "no information" sentence.
- **Robot hands now**: Dex3-1 output added (`--hands dex3`) next to Inspire.
- AI service total: ~4.3 GB GPU (TTS + LLM + clone); since 2026-11-13 ~5.0 GB with speech recognition (2.2, Typhoon
  Whisper Turbo), shared 8 GB laptop GPU; `NO_LLM=1` saves 3 GB, `NO_ASR=1` 1.7 GB.

GPU use (TTS, LLM, MediaPipe on CPU) is short and mostly offline. Coordinate with the other sessions before long jobs
(shared 8 GB GPU).

## 9. Risks

- **5-finger hand availability**: the plan assumes Inspire RH56 hands on Unitree's Inspire DDS service. If the
  robot still has Dex3-1, the hand poses are approximated with 3 fingers until the hands are changed.
- **Balance with arm motion** under Unitree's controller: speed scale, reach envelope, sim mode B, first real
  tests slow with a person at the remote.
- **Wai palm contact from monocular video** is not exact; the IK correction step handles it.
- **Offline Thai TTS/LLM quality on 8 GB** (rare words, product names, answer quality): pronunciation overrides,
  pre-rendered fixed texts, answers grounded in `shop.json`, operator preview before speaking.
- **Robot speaker latency/volume** in a noisy shop: measured in S7. The fallback is an external speaker at the
  entrance with the same timeline.

## Open questions

None at the moment (answered 2026-11-09: Inspire RH56 hands, built-in TTS voice first with upload later, this
laptop, simulation first). Before S7: confirm the hands fitted on the robot and the Inspire DDS topic names.
