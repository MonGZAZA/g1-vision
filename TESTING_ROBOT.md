# exp05 on the real G1: test procedure (S7)

Version 2026-11-10 (Dex3-1 hands, rehearsed in the Dex3 sim; Dex3 wai with the 35° hand turn, 2026-11-13). First hardware tests of the exp05 greeter: upper body only, through `rt/arm_sdk`, while
**Unitree's own controller keeps the robot standing**. Do the steps in order. A step passes only when all its
criteria are met. Any abort criterion means **E-STOP in the UI first, then analyse**. After any code or parameter
change, start again from step 1.

## 1. What exp05 does and does not do on the robot

- It does: `rt/arm_sdk` (waist yaw + both arms, weight ramped 0 → 1 over 1.5 s), the hand topic (Dex3-1:
  `rt/dex3/{left,right}/cmd`; Inspire RH56: `rt/inspire/cmd`), and the robot speaker (`AudioClient.PlayStream`).
- It does not: walk, touch the legs, `rt/lowcmd`, debug mode, damping. **E-STOP in the UI = hand the arms back
  to Unitree's controller over 1.5 s** (the robot keeps standing). The real emergency stop is the **Unitree remote**.
- The waist roll/pitch are locked on our robot; exp05 never commands them.

## 2. Roles and rules

| Role | Duties |
|---|---|
| Operator (laptop) | runs `run_robot.sh` and the UI, presses **E-STOP** (red button) |
| Spotter (next to the robot) | holds the **Unitree remote** (damping / e-stop), keeps people out of arm reach, calls "STOP" |

1. Two people for every powered test.
2. Nobody within **1 m of the arms** (arm reach is 0.65 m from the shoulder; pointing sweeps sideways).
3. The robot stands on a flat floor in Unitree's normal standing mode, battery above 30 %. A gantry or harness is
   recommended for steps 2–4 (arm motion shifts the balance a little; the sim's balance margin stays ≥ 3 cm).
4. Anyone may call "STOP". The operator presses E-STOP; if the robot does not respond, the spotter uses the remote.
5. Only one program may publish `rt/arm_sdk`. Close exp02/exp04 servers and Unitree's app arm control.

## 3. Setup

Rehearse first in the sim, which now has the same Dex3-1 hands and hand topics as the robot (`./run_sim.sh`,
default `HANDS=dex3`). **Stop the sim stack before `run_robot.sh`**: both use port 8100 (run_robot.sh refuses to
start while 8100 is in use).

```bash
nmcli con up g1-wired                       # laptop 192.168.123.222 on enx00e04c80fe28
ping -c 3 192.168.123.161                   # robot motion computer
cd ~/robotics-simulation/exp05 && ./run_robot.sh           # HANDS=dex3 (default) or HANDS=inspire
```

Open http://localhost:8100. Audio output: choose **robot** in the header. Check the header: `ROBOT · state ok`.

## 4. Steps

| Step | Do | Pass | Abort |
|---|---|---|---|
| 1. Link, read-only | Start the server. Do not take over. Watch the joint table, hands, AI pill | Joint angles change when the spotter moves an arm gently (robot in a mode that allows it) or match the standing pose; hands report values; `AI TTS+LLM` | no state, wrong joint signs |
| 2. Take-over / release at speed 0.3 | Speed slider 0.3. **Take over arms** → wait for weight 1.00 → **Release to Unitree** | Weight ramps in 1.5 s, the arms do not jump either way | any jump, buzzing, the robot steps |
| 3. Test poses at 0.3 | Take over, `turn_left_15`, `turn_right_15`, `left_forward`, `right_forward`, `rest` | Each pose reached smoothly; torso turn ±15° without stepping | stepping, a joint at its limit, sway |
| 4. E-STOP check | During `left_forward`: **E-STOP** | Arms go back to Unitree's posture within ~1.5 s, robot keeps standing | robot sags (that would be damping: never expected) |
| 5. Hands | Take over, pose `left_forward` (hands away from the hips), then hand buttons open / flat / relaxed / closed on that hand; end with **relaxed** before `rest` | Fingers follow; flat and relaxed = thumb folded across the palm; no finger touches another | finger motor fault, overheating, the open thumb touching the hip at rest |
| 6. Pointing at 0.5, then 1.0 | Mockup shop (or the real one): point at a near (New Arrivals), a side (Cafe, Restrooms) and a straight-ahead booth | Arm and torso as in the sim; the spotter confirms the hand points at the booth; the robot does not step | stepping, wobble |
| 7. Speaker + latency + volume | Audio output **robot**. Volume slider (Scripts & voice → Voice) to a comfortable level from ~1.5 m with the shop's normal noise. Point at a booth with Thai, then English | Voice from the robot; it starts as the arm arrives. If it is late, set `audio_latency_s` (POST /api/settings) until the voice and the caption start together | no sound: check `AudioClient` (robot audio service), volume |
| 8. Wai + introduce, service, Q&A | Speed 0.5, then 1.0. **Sawasdee + introduce** TH and EN; **Introduce service**; ask a FAQ and an LLM question, **Speak + point** | Wai with each hand turned 35° inward (Dex3, `gestures.WAI_HAND_TURN_DEG`): finger edges ~3.7 cm apart in front, thumbs folded, thumb bases toward the chest, the hands never touching (clip `clips/wai_dex3.json`, nearest gap 1.4 cm in the sim), "สวัสดีค่ะ" in the hold, gestures while speaking, idle between actions; no stepping | stepping, a hand touching the body, the thumbs pressing on each other |
| 9. Remote monitor | Open http://<laptop IP>:8100/monitor from another machine on the network | Caption and status follow live | — |

Record for each step: date, speed, pass/fail, notes, video. After step 9: full-speed run of all actions twice.

## 5. Known differences from the simulation

- The sim pins the pelvis (Unitree's standing is not simulated) and reports a ZMP balance margin; on the robot,
  watch the feet. Slower speed (slider) is the fix for any balance problem.
- Arms sag 1–2° under gravity in the sim's PD; the real arm drivers may differ (pointing tolerance is 5°).
- Dex3-1 (found in the Dex3 sim, 2026-11-10): an OPEN Dex3 thumb sticks out 8.8 cm on the palm side. So with
  Dex3, "flat" and "relaxed" fold the thumb across the palm (5 mm clear of the fingers). Otherwise the two thumbs
  hit each other in the wai and touch the hips with the arms hanging. exp05's rest pose has the shoulder roll
  at 0.18 rad (Inspire 0.12) for ~2 cm hip clearance, so after take-over the arms move ~3-4° outward from
  Unitree's posture.
- "open" with the arms at rest puts the thumbs on the hips: only open the hands with the arms forward.
- The wai and gestures are solved on the Dex3 model (`model/g1_dex3.xml`, palm frame identical to the Inspire
  one), so the arm angles fit the Dex3 hand: palm gap 2.0 cm, left/right symmetric.
- Robot speaker latency is a first guess (0.25 s) until measured in step 7.
