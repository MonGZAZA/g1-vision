"""Simulated G1 (29 DoF, locked waist, Dex3-1 or Inspire hands: EXP05_HANDS) for exp05, on the robot's DDS interface.

What it stands in for: the real G1 in Unitree's normal mode, standing. Unitree's controller holds the legs and, unless
the upper body is taken over, the arms. exp05 never takes the legs; it drives the waist + arms through ``rt/arm_sdk``
(unitree_hg LowCmd_, motors 12..28; motor_cmd[29].q = weight 0..1) and the hands through ``rt/dex3/<side>/cmd``
(Dex3-1) or ``rt/inspire/cmd`` (Inspire), like the real hands.

  Unitree standing   legs + arms held by PD at a standing pose (kp/kd below). This is NOT Unitree's balance
                     controller: the pelvis is pinned at standing height (legs PD on a fixed pelvis).
  rt/arm_sdk         like the real robot: for motors 12..28, target = w * arm_sdk command + (1 - w) * Unitree's own
                     posture, gains blended the same way (w = clip(motor_cmd[29].q, 0, 1)). The two locked waist
                     motors are ignored. If arm_sdk stops for > 0.5 s, w decays to 0 over 1 s (safety, like letting go).
  rt/dex3/<side>/cmd unitree_hg HandCmd_, 7 motors (thumb 0-2, middle 0-1, index 0-1), q in rad; the sim servos each
                     finger joint to q (clipped to the joint range) while the motor's kp > 0.
  rt/dex3/<side>/state  unitree_hg HandState_, 7 motor q, 50 Hz.
  rt/inspire/cmd     unitree_go MotorCmds_, 12 entries (right 0-5, left 6-11; per hand: pinky, ring, middle, index,
                     thumb bend, thumb rotation), q = 1 open .. 0 closed. Fingers are simulated physically (the
                     URDF's mimic joints are equalities).
  rt/lowstate        unitree_hg LowState_: 29 motors (locked waist slots q = 0), pelvis IMU, 250 Hz by default.
  rt/inspire/state   unitree_go MotorStates_, 12 entries, same 0..1 scale, 50 Hz.

Balance indicator (instead of simulating Unitree's balance controller, which we do not have): every 20 ms the sim
computes the whole-body centre of mass (COM) and the ZMP (cart-table model: ZMP = COM_xy - h / g * COM_acc_xy) and its
margin to the edge of the feet's support polygon (+ = inside). Published as JSON on the sim-only topic
``rt/sim/balance`` (std_msgs String). Arm motion that keeps the margin well above ~3 cm does not need balance help;
Unitree's controller has to work harder below that. (A free-standing mode with plain leg PD fell over at once: a
33 kg robot needs ~230 N m/rad ankle stiffness, so it would only test our stand-in, not Unitree's controller.)

Scene: with --shop shops/<name>/shop.json the booths are added as translucent boxes around the robot (robot frame:
the robot stands at the origin facing +x), so the viewer shows what it points at.

    .venv/bin/python deploy/sim_robot.py [--headless] [--shop ../shops/mockup/shop.json]
"""

import argparse
import json
import math
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (unitree_go_msg_dds__MotorState_, unitree_hg_msg_dds__HandState_,
                                        unitree_hg_msg_dds__LowState_)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_

from robot_model import (ARM_REST, ARM_SDK_MOTORS, DEX3_MOTORS, HAND_MODEL, HAND_POSES, INSPIRE_DOFS, LOCKED, MODEL,
                         MOTORS, NUM_MOTOR, STAND_HEIGHT, STAND_LEGS, WEIGHT_SLOT, inspire_to_dex3, inspire_to_joint,
                         joint_to_inspire, lowest_point, shop_booths_robot_frame)

DOMAIN = 5
KP_LEG = {"hip": 150.0, "knee": 200.0, "ankle": 60.0}
KD_LEG = {"hip": 4.0, "knee": 6.0, "ankle": 2.0}
KP_UPPER, KD_UPPER = 80.0, 2.0  # Unitree's own posture hold for waist + arms (when w = 0)
ARM_SDK_TIMEOUT, ARM_SDK_FADE = 0.5, 1.0


def support_polygon(m, d):
    """Convex hull (counter-clockwise), xy, of the feet's contact spheres (4 per foot in the model)."""
    pts = [d.geom_xpos[g][:2].copy() for g in range(m.ngeom)
           if m.body(m.geom_bodyid[g]).name.endswith("ankle_roll_link") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    pts = sorted({(round(float(p[0]), 4), round(float(p[1]), 4)) for p in pts})
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_margin(poly, p):
    """Signed distance from p to the polygon edge (+ inside), m."""
    if len(poly) < 3:
        return float("nan")
    best = math.inf
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        ex, ey = x2 - x1, y2 - y1
        n = math.hypot(ex, ey)
        best = min(best, (ex * (p[1] - y1) - ey * (p[0] - x1)) / n)  # left of a CCW edge = inside
    return best


def build(shop):
    spec = mujoco.MjSpec.from_file(str(MODEL))
    if shop:
        for b in shop_booths_robot_frame(shop):
            body = spec.worldbody.add_body(name=f"booth_{b['id']}", pos=[b["cx"], b["cy"], 0.5],
                                           quat=[math.cos(b["yaw"] / 2), 0, 0, math.sin(b["yaw"] / 2)])
            rgba = b["rgba"][:3] + [0.35]
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[b["w"] / 2, b["h"] / 2, 0.5], rgba=rgba,
                          contype=0, conaffinity=0)
            spec.worldbody.add_site(name=f"aim_{b['id']}", pos=b["aim"], size=[0.08] * 3, rgba=[1, 0.1, 0.1, 0.8])
    m = spec.compile()
    m.vis.map.zfar = 80.0
    return m


class SimRobot:
    def __init__(self, args):
        self.args = args
        shop = json.loads(Path(args.shop).read_text()) if args.shop else None
        self.m = m = build(shop)
        self.d = d = mujoco.MjData(m)
        self.act = {}  # motor index -> actuator id (locked waist slots have none)
        for i, name in enumerate(MOTORS):
            if name not in LOCKED:
                self.act[i] = m.actuator(name.replace("_joint", "")).id
        self.qadr = {i: m.jnt_qposadr[m.actuator_trnid[a, 0]] for i, a in self.act.items()}
        self.dadr = {i: m.jnt_dofadr[m.actuator_trnid[a, 0]] for i, a in self.act.items()}
        self.dex3 = HAND_MODEL == "dex3"
        dofs = DEX3_MOTORS if self.dex3 else INSPIRE_DOFS
        self.hand_act = {(s, k): m.actuator(f"{s}_hand_{k}").id for s in ("left", "right") for k in dofs}
        self.hand_qadr = {sk: m.jnt_qposadr[m.actuator_trnid[a, 0]] for sk, a in self.hand_act.items()}
        self.hand_range = {sk: m.jnt_range[m.actuator_trnid[a, 0]].copy() for sk, a in self.hand_act.items()}
        # rad; start with the "relaxed" pose (thumb folded: an open Dex3 thumb touches the hips at the rest pose)
        self.dex3_cmd = {s: inspire_to_dex3(HAND_POSES["relaxed"], s) for s in ("left", "right")}

        # Unitree's own posture (what the robot holds without arm_sdk)
        self.own_q = np.zeros(NUM_MOTOR)
        self.own_kp = np.zeros(NUM_MOTOR)
        self.own_kd = np.zeros(NUM_MOTOR)
        for i, name in enumerate(MOTORS):
            for k, v in STAND_LEGS.items():
                if k in name:
                    self.own_q[i] = v
            for k, v in ARM_REST.items():
                if name == k:
                    self.own_q[i] = v
            part = next((p for p in KP_LEG if p in name), None)
            self.own_kp[i] = KP_LEG[part] if part else KP_UPPER
            self.own_kd[i] = KD_LEG[part] if part else KD_UPPER
        for i, a in self.qadr.items():
            d.qpos[a] = self.own_q[i]
        d.qpos[2] = STAND_HEIGHT
        d.qpos[3] = 1.0
        for sk, a in self.hand_qadr.items():
            d.qpos[a] = (self.dex3_cmd[sk[0]][DEX3_MOTORS.index(sk[1])] if self.dex3
                         else inspire_to_joint(1.0, self.hand_range[sk]))
        mujoco.mj_forward(m, d)
        # stand on the floor: lowest foot point at z = 0
        d.qpos[2] -= lowest_point(m, d)
        mujoco.mj_forward(m, d)
        self.pin = d.qpos[:7].copy()  # pelvis held kinematically at this pose

        self.lock = threading.Lock()
        self.arm_cmd = None  # (q, kp, kd, dq, tau) per slot, for all 29 slots
        self.arm_w_cmd = 0.0
        self.arm_time = 0.0
        self.w = 0.0
        self.hand_cmd = np.ones(12)  # Inspire scale, 1 = open
        self.imu = {n: m.sensor_adr[m.sensor(n).id] for n in ("imu_quat", "imu_gyro", "imu_acc")}

        ChannelFactoryInitialize(args.domain, args.net)
        self.state = unitree_hg_msg_dds__LowState_()
        self.pub = ChannelPublisher("rt/lowstate", LowState_)
        self.pub.Init()
        if self.dex3:
            self.dex3_pub, self.dex3_sub = {}, {}
            for s in ("left", "right"):
                self.dex3_pub[s] = ChannelPublisher(f"rt/dex3/{s}/state", HandState_)
                self.dex3_pub[s].Init()
                self.dex3_sub[s] = ChannelSubscriber(f"rt/dex3/{s}/cmd", HandCmd_)
                self.dex3_sub[s].Init(lambda msg, s=s: self.on_dex3(s, msg), 10)
        else:
            self.hand_pub = ChannelPublisher("rt/inspire/state", MotorStates_)
            self.hand_pub.Init()
        self.sub = ChannelSubscriber("rt/arm_sdk", LowCmd_)
        self.sub.Init(self.on_arm_sdk, 10)
        self.balance_pub = ChannelPublisher("rt/sim/balance", String_)
        self.balance_pub.Init()
        if not self.dex3:
            self.hand_sub = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
            self.hand_sub.Init(self.on_hand, 10)
        for _ in range(200):  # settle the feet on the floor, then take the support polygon from the foot contacts
            self.step()
        self.support = support_polygon(m, d)
        self.balance = {}
        self._com_v = None

    def on_arm_sdk(self, msg):
        mc = msg.motor_cmd
        arr = np.array([[mc[i].q, mc[i].kp, mc[i].kd, mc[i].dq, mc[i].tau] for i in range(NUM_MOTOR)], dtype=float)
        w = float(np.clip(mc[WEIGHT_SLOT].q, 0.0, 1.0))
        with self.lock:
            self.arm_cmd, self.arm_w_cmd, self.arm_time = arr, w, time.monotonic()

    def on_hand(self, msg):
        v = np.array([c.q for c in msg.cmds[:12]], dtype=float)
        with self.lock:
            self.hand_cmd = np.clip(v, 0.0, 1.0)

    def on_dex3(self, side, msg):
        mc = msg.motor_cmd
        with self.lock:
            for i, k in enumerate(DEX3_MOTORS):
                if mc[i].kp > 0:  # kp 0 = motor limp: keep the last target
                    self.dex3_cmd[side][i] = float(np.clip(mc[i].q, *self.hand_range[(side, k)]))

    def step(self):
        m, d = self.m, self.d
        with self.lock:
            cmd = self.arm_cmd
            age = time.monotonic() - self.arm_time
            w_cmd = self.arm_w_cmd if cmd is not None else 0.0
            hand = self.hand_cmd.copy()
            dex3 = {s: v.copy() for s, v in self.dex3_cmd.items()}
        if cmd is not None and age > ARM_SDK_TIMEOUT:  # stream stopped: fade back to Unitree's posture
            w_cmd = 0.0
            self.w = max(0.0, self.w - m.opt.timestep / ARM_SDK_FADE)
        else:
            self.w = w_cmd
        w = self.w
        for i, a in self.act.items():
            q, kp, kd, dq, tau = self.own_q[i], self.own_kp[i], self.own_kd[i], 0.0, 0.0
            if i in ARM_SDK_MOTORS and cmd is not None and w > 0:
                q = w * cmd[i, 0] + (1 - w) * q
                kp = w * cmd[i, 1] + (1 - w) * kp
                kd = w * cmd[i, 2] + (1 - w) * kd
                dq, tau = w * cmd[i, 3], w * cmd[i, 4]
            d.ctrl[a] = kp * q + kd * dq + tau
            m.actuator_biasprm[a, 1] = -kp
            m.actuator_biasprm[a, 2] = -kd
        for (s, k), a in self.hand_act.items():
            if self.dex3:
                tgt = dex3[s][DEX3_MOTORS.index(k)]
            else:
                idx = (0 if s == "right" else 6) + INSPIRE_DOFS.index(k)
                tgt = inspire_to_joint(hand[idx], self.hand_range[(s, k)])
            d.ctrl[a] = tgt  # position servo (gain = -bias kp)
        mujoco.mj_step(m, d)
        if self.pin is not None:
            d.qpos[:7] = self.pin
            d.qvel[:6] = 0.0

    def publish(self, tick):
        d, s = self.d, self.state
        for i in range(NUM_MOTOR):
            ms = s.motor_state[i]
            if i in self.act:
                ms.q = float(d.qpos[self.qadr[i]])
                ms.dq = float(d.qvel[self.dadr[i]])
                ms.tau_est = float(d.actuator_force[self.act[i]])
            else:
                ms.q = ms.dq = ms.tau_est = 0.0
            ms.temperature = [35, 35]
        q = d.sensordata[self.imu["imu_quat"]: self.imu["imu_quat"] + 4]
        s.imu_state.quaternion = [float(v) for v in q]
        s.imu_state.gyroscope = [float(v) for v in d.sensordata[self.imu["imu_gyro"]: self.imu["imu_gyro"] + 3]]
        s.imu_state.accelerometer = [float(v) for v in d.sensordata[self.imu["imu_acc"]: self.imu["imu_acc"] + 3]]
        s.tick = tick
        self.pub.Write(s)

    def update_balance(self, dt):
        m, d = self.m, self.d
        mujoco.mj_subtreeVel(m, d)
        com, v = d.subtree_com[1].copy(), d.subtree_linvel[1].copy()
        acc = np.zeros(3) if self._com_v is None else (v - self._com_v) / dt
        self._com_v = v
        zmp = com[:2] - com[2] / 9.81 * acc[:2]
        self.balance = {"com": np.round(com, 4).tolist(), "zmp": np.round(zmp, 4).tolist(),
                        "com_margin": round(polygon_margin(self.support, com[:2]), 4),
                        "zmp_margin": round(polygon_margin(self.support, zmp), 4), "t": round(d.time, 3)}
        self.balance_pub.Write(String_(data=json.dumps(self.balance)))

    def publish_hands(self):
        d = self.d
        if self.dex3:
            for s in ("left", "right"):
                st = unitree_hg_msg_dds__HandState_()
                for i, k in enumerate(DEX3_MOTORS):
                    st.motor_state[i].q = float(d.qpos[self.hand_qadr[(s, k)]])
                self.dex3_pub[s].Write(st)
            return
        states = [unitree_go_msg_dds__MotorState_() for _ in range(12)]
        for (side, k), a in self.hand_qadr.items():
            idx = (0 if side == "right" else 6) + INSPIRE_DOFS.index(k)
            states[idx].q = float(joint_to_inspire(d.qpos[a], self.hand_range[(side, k)]))
        self.hand_pub.Write(MotorStates_(states=states))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", default="lo")
    p.add_argument("--domain", type=int, default=DOMAIN)
    p.add_argument("--state-hz", type=float, default=250.0, help="rt/lowstate rate")
    p.add_argument("--shop", default="", help="shop.json: show the booths around the robot")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    sim = SimRobot(args)
    dt = sim.m.opt.timestep
    every = max(1, round(1 / (args.state_hz * dt)))
    print(f"[SIM] G1 lock-waist + {HAND_MODEL} hands on DDS domain {args.domain} ({args.net}); pelvis pinned; "
          f"support polygon {len(sim.support)} corners; lowstate {1 / (every * dt):.0f} Hz", flush=True)
    data_lock = threading.Lock()
    running = threading.Event()
    running.set()

    def physics():
        n, t_next = 0, time.perf_counter()
        rtf_wall, rtf_sim = time.perf_counter(), sim.d.time
        while running.is_set():
            with data_lock:
                sim.step()
                if n % every == 0:
                    sim.publish(n)
                if n % 10 == 0:
                    sim.publish_hands()
                    sim.update_balance(10 * dt)
            n += 1
            now = time.perf_counter()
            if now - rtf_wall > 30.0:
                print(f"[SIM] real-time factor {(sim.d.time - rtf_sim) / (now - rtf_wall):.2f}, "
                      f"arm_sdk weight {sim.w:.2f}, pelvis z {sim.d.qpos[2]:.3f}", flush=True)
                rtf_wall, rtf_sim = now, sim.d.time
            t_next += dt
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.1:
                t_next = time.perf_counter()

    thread = threading.Thread(target=physics, name="physics", daemon=True)
    thread.start()
    try:
        if args.headless:
            while thread.is_alive():
                time.sleep(0.5)
        else:
            d_view = mujoco.MjData(sim.m)
            with data_lock:
                mujoco.mj_copyData(d_view, sim.m, sim.d)
            with mujoco.viewer.launch_passive(sim.m, d_view) as viewer:
                viewer.cam.lookat[:] = [0.6, 0, 0.9]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 200, -15
                while viewer.is_running():
                    with data_lock:
                        mujoco.mj_copyData(d_view, sim.m, sim.d)
                    viewer.sync()
                    time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    running.clear()
    thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
