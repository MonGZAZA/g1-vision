"""S0 check: drive the sim robot through rt/arm_sdk + the hand topic like exp05 will drive the real robot.
Hands from EXP05_HANDS (default dex3: rt/dex3/<side>/cmd|state; inspire: rt/inspire/cmd|state).

Starts deploy/sim_robot.py (headless, own DDS domain) and checks:
  1. rt/lowstate arrives at the expected rate; locked waist slots report 0
  2. weight ramp 0 -> 1 over 1 s with the arm targets = measured pose (no jump)
  3. a target pose (left arm raised forward, waist yaw 15 deg) is reached within 3 deg
  4. hands: close the right hand, open it; the state follows (Dex3: also the model's ranges = DEX3_FAR)
  5. weight 1 -> 0: the arms go back to Unitree's posture
  6. stream stop with weight 1: the sim fades back on its own (safety)
  7. balance indicator (rt/sim/balance): support polygon found, COM/ZMP inside with margin at rest and in the pose
Usage: .venv/bin/python tools/sim_check.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from robot_model import (ARM_REST, ARM_SDK_MOTORS, DEX3_FAR, DEX3_MOTORS, HAND_MODEL, HAND_POSES, LOCKED, MODEL, MOTORS,  # noqa: E402
                         NUM_MOTOR, WEIGHT_SLOT, dex3_to_inspire, inspire_to_dex3)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import (unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__HandCmd_,  # noqa: E402
                                        unitree_hg_msg_dds__LowCmd_)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

DOMAIN = 51  # private test domain
G1 = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self):
        ChannelFactoryInitialize(DOMAIN, "lo")
        self.q = None
        self.n = 0
        self.hand = None
        ChannelSubscriber("rt/lowstate", LowState_).Init(self.on_state, 10)
        self.dex3 = HAND_MODEL == "dex3"
        if self.dex3:
            self.dex3_q = {"left": None, "right": None}
            self.dpub = {}
            for sd in ("left", "right"):
                ChannelSubscriber(f"rt/dex3/{sd}/state", HandState_).Init(lambda m, sd=sd: self.on_dex3(sd, m), 10)
                self.dpub[sd] = ChannelPublisher(f"rt/dex3/{sd}/cmd", HandCmd_)
                self.dpub[sd].Init()
        else:
            ChannelSubscriber("rt/inspire/state", MotorStates_).Init(
                lambda m: setattr(self, "hand", np.array([s.q for s in m.states[:12]])), 10)
        self.balance_log = []
        ChannelSubscriber("rt/sim/balance", String_).Init(lambda m: self.balance_log.append(json.loads(m.data)), 10)
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        if not self.dex3:
            self.hpub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
            self.hpub.Init()
        self.crc = CRC()

    def on_dex3(self, sd, m):
        self.dex3_q[sd] = np.array([s.q for s in m.motor_state[:7]])
        if all(v is not None for v in self.dex3_q.values()):  # Inspire values, as the arm channel reports them
            self.hand = np.concatenate([dex3_to_inspire(self.dex3_q["right"], "right"),
                                        dex3_to_inspire(self.dex3_q["left"], "left")])

    def on_state(self, m):
        self.q = np.array([m.motor_state[i].q for i in range(NUM_MOTOR)])
        self.imu = list(m.imu_state.quaternion)
        self.n += 1

    def send(self, q, w, kp=60.0, kd=1.5):
        c = unitree_hg_msg_dds__LowCmd_()
        for i in ARM_SDK_MOTORS:
            c.motor_cmd[i].q, c.motor_cmd[i].kp, c.motor_cmd[i].kd = float(q[i]), kp, kd
        c.motor_cmd[WEIGHT_SLOT].q = float(w)
        c.crc = self.crc.Crc(c)
        self.pub.Write(c)

    def hand_cmd(self, v12):
        if self.dex3:
            for sd, sl in (("right", slice(0, 6)), ("left", slice(6, 12))):
                c = unitree_hg_msg_dds__HandCmd_()
                for i, (mc, q) in enumerate(zip(c.motor_cmd, inspire_to_dex3(np.asarray(v12)[sl], sd))):
                    mc.mode, mc.q, mc.kp, mc.kd = (i & 0x0F) | (1 << 4), float(q), 1.5, 0.1
                self.dpub[sd].Write(c)
            return
        cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
        for c, v in zip(cmds, v12):
            c.q = float(v)
        self.hpub.Write(MotorCmds_(cmds=cmds))


def run(c, q, w, secs, hz=50):
    for _ in range(int(secs * hz)):
        c.send(q, w)
        time.sleep(1 / hz)


def main():
    sim = subprocess.Popen([sys.executable, "deploy/sim_robot.py", "--headless", "--domain", str(DOMAIN)], cwd=G1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    results = []

    def check(name, ok, info=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

    try:
        if HAND_MODEL == "dex3":
            import mujoco
            m = mujoco.MjModel.from_xml_path(str(MODEL))
            bad = [(sd, k) for sd in ("left", "right") for i, k in enumerate(DEX3_MOTORS)
                   if DEX3_FAR[sd][i] != 0 and abs(min(m.joint(f"{sd}_hand_{k}_joint").range, key=lambda r: abs(r - DEX3_FAR[sd][i])) - DEX3_FAR[sd][i]) > 1e-3]
            check("Dex3 ranges in the model = DEX3_FAR", not bad, str(bad) if bad else "")
        c = Client()
        t0 = time.time()
        while c.q is None and time.time() - t0 < 30:
            time.sleep(0.1)
        n0 = c.n
        time.sleep(2.0)
        rate = (c.n - n0) / 2.0
        check("lowstate rate", rate > 200, f"{rate:.0f} Hz")
        lk = [abs(c.q[MOTORS.index(n)]) for n in LOCKED]
        check("locked waist slots = 0", max(lk) == 0.0)
        rest = c.q.copy()
        rest_ok = all(abs(rest[MOTORS.index(k)] - v) < 0.05 for k, v in ARM_REST.items())
        check("Unitree posture held at start", rest_ok)

        q = rest.copy()
        for k in range(50):  # ramp the weight over 1 s, targets = current pose
            c.send(q, (k + 1) / 50)
            time.sleep(0.02)
        drift = np.max(np.abs(c.q[ARM_SDK_MOTORS] - rest[ARM_SDK_MOTORS]))
        check("take-over without jump", drift < 0.05, f"max change {np.degrees(drift):.1f} deg")

        tgt = rest.copy()
        tgt[MOTORS.index("waist_yaw_joint")] = np.radians(15)
        tgt[MOTORS.index("left_shoulder_pitch_joint")] = -1.2
        tgt[MOTORS.index("left_elbow_joint")] = 1.2
        for k in range(100):  # 2 s linear move
            run(c, rest + (tgt - rest) * (k + 1) / 100, 1.0, 0.02)
        run(c, tgt, 1.0, 1.0)
        err = np.degrees(np.max(np.abs(c.q[ARM_SDK_MOTORS] - tgt[ARM_SDK_MOTORS])))
        check("target pose reached", err < 3.0, f"max error {err:.1f} deg")

        closed = np.ones(12)
        closed[:6] = 0.0  # right hand closed
        for _ in range(50):
            c.hand_cmd(closed)
            c.send(tgt, 1.0)
            time.sleep(0.02)
        h = c.hand
        check("right hand closes", h is not None and np.max(np.abs(h[:4] - 0)) < 0.15 and np.min(h[6:]) > 0.9,
              f"right {np.round(h[:6], 2) if h is not None else None}")
        for _ in range(50):
            c.hand_cmd(np.ones(12))
            c.send(tgt, 1.0)
            time.sleep(0.02)
        check("hand opens", c.hand is not None and np.min(c.hand) > 0.85, f"min {np.min(c.hand):.2f}")
        for _ in range(50):  # back to "relaxed" before the arms go down (an open Dex3 thumb touches the hips)
            c.hand_cmd(np.array(HAND_POSES["relaxed"] * 2, float))
            c.send(tgt, 1.0)
            time.sleep(0.02)

        for k in range(50):  # release over 1 s
            c.send(tgt, 1 - (k + 1) / 50)
            time.sleep(0.02)
        time.sleep(1.5)
        back = np.degrees(np.max(np.abs(c.q[ARM_SDK_MOTORS] - rest[ARM_SDK_MOTORS])))
        check("release: back to Unitree posture", back < 3.0, f"max diff {back:.1f} deg")

        smooth_end = len(c.balance_log)
        run(c, tgt, 1.0, 3.0)  # take over again (hard), then stop the stream
        time.sleep(0.3)
        held = np.degrees(np.max(np.abs(c.q[ARM_SDK_MOTORS] - tgt[ARM_SDK_MOTORS])))
        time.sleep(3.0)
        back = np.degrees(np.max(np.abs(c.q[ARM_SDK_MOTORS] - rest[ARM_SDK_MOTORS])))
        check("stream stop: sim fades back", held < 5 and back < 3.0, f"held {held:.1f}, then {back:.1f} deg")
        b = c.balance_log
        rest_m = [x["zmp_margin"] for x in b if x["t"] < 2.0]
        smooth = [x["zmp_margin"] for x in b[:smooth_end]]
        hard = [x["zmp_margin"] for x in b[smooth_end:]]
        check("balance: ZMP inside the feet during smooth motion", len(smooth) > 50 and min(smooth) > 0.02,
              f"rest {np.mean(rest_m) * 100:.1f} cm, min {min(smooth) * 100:.1f} cm" if smooth else "no data")
        print(f"       (info) instant 0 -> 1 take-over jump: ZMP margin min {min(hard) * 100:.1f} cm: "
              "the motion layer must never do that")
    finally:
        sim.terminate()
        out = sim.communicate(timeout=5)[0]
    print(out.strip().splitlines()[-1] if out.strip() else "")
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
