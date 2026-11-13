"""The only path from exp05 to the robot's upper body: rt/arm_sdk (waist yaw + both arms) and the hand topic
(Dex3-1: rt/dex3/{left,right}/cmd; Inspire RH56: rt/inspire/cmd).

Safety rules live here, so no motion code can bypass them:
  - take-over: the weight ramps 0 -> 1 over TAKEOVER_S with the targets = the measured pose (no jump);
  - release:   the targets are held, the weight ramps 1 -> 0 over RELEASE_S (Unitree's posture takes back over);
  - E-STOP:    release, nothing else (never damping: the robot stands under Unitree's controller);
  - every target is clipped to the joint limits minus a margin and rate-limited to MAX_SPEED * speed_scale;
  - targets older than STALE_S (the motion layer stopped) -> automatic release;
  - no state from the robot (rt/lowstate) for STALE_S -> release, and no take-over without state.
The loop runs at RATE_HZ in its own thread; the motion layer only sets goals (set_goal / set_hands).
"""

import threading
import time

import mujoco
import numpy as np

from robot_model import (ARM_SDK_MOTORS, HAND_POSES, HANDS, INSPIRE_DOFS, MODEL, MOTORS, NUM_MOTOR,  # noqa: F401
                         WEIGHT_SLOT, dex3_to_inspire, inspire_to_dex3)
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__HandCmd_,
                                        unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__MotorCmd_)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

RATE_HZ = 100
TAKEOVER_S, RELEASE_S = 1.5, 1.5
STALE_S = 0.5
MAX_SPEED = 2.0  # rad/s at speed scale 1
LIMIT_MARGIN = 0.05  # rad inside the model's joint limits
KP = {"waist_yaw": 120.0, "shoulder": 60.0, "elbow": 50.0, "wrist": 20.0}  # Unitree arm_sdk example uses 60 / 1.5
KD = {"waist_yaw": 3.0, "shoulder": 1.5, "elbow": 1.5, "wrist": 0.8}
HAND_RATE = 2.0  # Inspire units (0..1) per second

# Dex3-1 mapping (Inspire values <-> 7 motors per hand): robot_model.inspire_to_dex3; gains from Unitree's dex3
# example (as in exp02 hands.py).
DEX3_KP, DEX3_KD = 1.5, 0.1


def _gain(table, name):
    return next(v for k, v in table.items() if k in name)


class ArmChannel:
    def __init__(self, hands=HANDS):
        """hands: "dex3" (Dex3-1), "inspire" (RH56) or "none" (sim and robot alike)."""
        m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.idx = list(ARM_SDK_MOTORS)
        self.names = [MOTORS[i] for i in self.idx]
        rng = np.array([m.jnt_range[m.joint(n).id] for n in self.names])
        self.lo, self.hi = rng[:, 0] + LIMIT_MARGIN, rng[:, 1] - LIMIT_MARGIN
        self.kp = np.array([_gain(KP, n) for n in self.names])
        self.kd = np.array([_gain(KD, n) for n in self.names])

        self.lock = threading.Lock()
        self.q_meas = None  # all 29 motors
        self.state_time = 0.0
        self.state_count = 0
        self.imu_quat = [1.0, 0.0, 0.0, 0.0]
        self.hand_meas = None
        self.mode = "released"  # released | takeover | active | releasing
        self.weight = 0.0
        self.target = None  # 15 upper joints, the rate-limited command
        self.goal = None  # 15 upper joints, from the motion layer
        self.goal_time = 0.0
        self.hand_goal = np.ones(12)
        self.hand_cmd = np.ones(12)
        self.hands_on = False  # hand commands are only sent after the first set_hands
        self.speed_scale = 0.5
        self.reason = ""
        self.crc = CRC()

        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on_state, 10)
        self.hand_type = hands
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        if hands == "inspire":
            ChannelSubscriber("rt/inspire/state", MotorStates_).Init(self._on_hand, 10)
            self.hpub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
            self.hpub.Init()
        elif hands == "dex3":
            self.dex3_meas = {"left": None, "right": None}
            self.dpub = {}
            for sd in ("left", "right"):
                ChannelSubscriber(f"rt/dex3/{sd}/state", HandState_).Init(lambda m, sd=sd: self._on_dex3(sd, m), 10)
                self.dpub[sd] = ChannelPublisher(f"rt/dex3/{sd}/cmd", HandCmd_)
                self.dpub[sd].Init()
        self.running = True
        threading.Thread(target=self._loop, name="arm_channel", daemon=True).start()

    # ---- robot state
    def _on_state(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(NUM_MOTOR)])
        with self.lock:
            self.q_meas, self.state_time = q, time.monotonic()
            self.imu_quat = list(msg.imu_state.quaternion)
            self.state_count += 1

    def _on_hand(self, msg):
        with self.lock:
            self.hand_meas = np.array([s.q for s in msg.states[:12]])

    def _on_dex3(self, side, msg):
        q = [m.q for m in msg.motor_state[:7]]
        with self.lock:
            self.dex3_meas[side] = dex3_to_inspire(q, side)
            if self.dex3_meas["left"] is not None and self.dex3_meas["right"] is not None:
                self.hand_meas = np.concatenate([self.dex3_meas["right"], self.dex3_meas["left"]])

    def upper_measured(self):
        with self.lock:
            return None if self.q_meas is None else self.q_meas[self.idx].copy()

    def state_ok(self):
        return self.q_meas is not None and time.monotonic() - self.state_time < STALE_S

    # ---- commands from the server / motion layer
    def takeover(self):
        with self.lock:
            if self.q_meas is None or time.monotonic() - self.state_time > STALE_S:
                self.reason = "no robot state"
                return False
            if self.mode in ("active", "takeover"):
                return True
            self.target = self.q_meas[self.idx].copy()
            self.goal, self.goal_time = self.target.copy(), time.monotonic()
            self.mode, self.reason = "takeover", ""
            return True

    def release(self, reason="operator"):
        with self.lock:
            if self.mode in ("takeover", "active"):
                self.mode, self.reason = "releasing", reason

    def set_goal(self, q15):
        """Upper-body goal (15 joints, ARM_SDK_MOTORS order); call at >= 1/STALE_S Hz while active."""
        with self.lock:
            self.goal = np.clip(np.asarray(q15, dtype=float), self.lo, self.hi)
            self.goal_time = time.monotonic()

    def set_hands(self, v12):
        """Inspire goal, 12 values (right 0-5, left 6-11), 1 = open."""
        with self.lock:
            if not self.hands_on and self.hand_meas is not None:
                self.hand_cmd = self.hand_meas.copy()  # start from the measured fingers (no jump)
            self.hand_goal = np.clip(np.asarray(v12, dtype=float), 0.0, 1.0)
            self.hands_on = True

    def status(self):
        with self.lock:
            return {"mode": self.mode, "weight": round(self.weight, 3), "speed_scale": self.speed_scale,
                    "state_ok": self.state_ok(), "reason": self.reason,
                    "q": None if self.q_meas is None else np.round(self.q_meas[self.idx], 4).tolist(),
                    "target": None if self.target is None else np.round(self.target, 4).tolist(),
                    "hands": None if self.hand_meas is None else np.round(self.hand_meas, 3).tolist(),
                    "imu_quat": [round(v, 4) for v in self.imu_quat]}

    # ---- 100 Hz loop
    def _loop(self):
        dt = 1.0 / RATE_HZ
        t_next = time.perf_counter()
        while self.running:
            self._tick(dt)
            t_next += dt
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()

    def _tick(self, dt):
        now = time.monotonic()
        with self.lock:
            if self.mode != "released":
                if now - self.state_time > STALE_S and self.mode != "releasing":
                    self.mode, self.reason = "releasing", "robot state stale"
                if self.mode == "active" and now - self.goal_time > STALE_S:
                    self.mode, self.reason = "releasing", "motion goals stopped"
                if self.mode == "takeover":
                    self.weight = min(1.0, self.weight + dt / TAKEOVER_S)
                    if self.weight >= 1.0:
                        self.mode = "active"
                        self.goal_time = now
                elif self.mode == "releasing":
                    self.weight = max(0.0, self.weight - dt / RELEASE_S)
                    if self.weight <= 0.0:
                        self.mode = "released"
                if self.mode == "active":
                    step = MAX_SPEED * self.speed_scale * dt
                    self.target = self.target + np.clip(self.goal - self.target, -step, step)
                target, weight, mode = self.target.copy(), self.weight, self.mode
            else:
                target, weight, mode = None, 0.0, "released"
            hs = HAND_RATE * self.speed_scale * dt
            self.hand_cmd = self.hand_cmd + np.clip(self.hand_goal - self.hand_cmd, -hs, hs)
            hand = self.hand_cmd.copy() if self.hands_on else None
        if target is not None:
            c = unitree_hg_msg_dds__LowCmd_()
            for k, i in enumerate(self.idx):
                mc = c.motor_cmd[i]
                mc.q, mc.dq, mc.tau, mc.kp, mc.kd = float(target[k]), 0.0, 0.0, float(self.kp[k]), float(self.kd[k])
            c.motor_cmd[WEIGHT_SLOT].q = float(weight)
            c.crc = self.crc.Crc(c)
            self.pub.Write(c)
            if mode == "released":  # last message carried weight 0
                with self.lock:
                    self.target = None
        if hand is None or self.hand_type == "none":
            return
        if self.hand_type == "dex3":
            for sd, sl in (("right", slice(0, 6)), ("left", slice(6, 12))):
                q7 = inspire_to_dex3(hand[sl], sd)
                c = unitree_hg_msg_dds__HandCmd_()
                c.motor_cmd = [unitree_hg_msg_dds__MotorCmd_() for _ in range(7)]
                for i, m in enumerate(c.motor_cmd):
                    m.mode = (i & 0x0F) | (1 << 4)  # RIS mode byte: motor id | enabled << 4
                    m.q, m.dq, m.tau, m.kp, m.kd = float(q7[i]), 0.0, 0.0, DEX3_KP, DEX3_KD
                self.dpub[sd].Write(c)
            return
        cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
        for cm, v in zip(cmds, hand):
            cm.q = float(v)
        self.hpub.Write(MotorCmds_(cmds=cmds))


