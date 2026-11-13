"""Pointing at a booth with an open palm (task 3): target geometry, arm choice, torso turn, elevation curve, IK.

Robot frame: the robot stands at the origin facing +x, y to its left, z up from the floor (robot_model.world_to_robot).

  1. aim point -> azimuth theta (+ = left) and horizontal distance d (from the robot's centre);
  2. arm: left if theta > +CENTER_DEG, right if theta < -CENTER_DEG, CENTER_ARM in between (never across the body);
  3. torso (= head, the G1 has no neck): psi = clamp(k * theta, +-psi_max), shop settings torso_k / torso_max_deg;
  4. ray from the turned shoulder: azimuth toward the aim point, elevation = elev_curve(d) + the aim point's height
     above the 1 m reference -> near booths clearly lower, far ones nearly horizontal;
  5. IK (damped least squares on the 7 arm joints, waist fixed at psi): wrist on the ray at REACH from the shoulder
     (elbow slightly bent), fingers along the ray, palm turned up by PALM_UP_DEG from facing inward;
  6. checks: ray azimuth / elevation error, joint limits, self-collision of the pointing arm.

Hand frame of the Inspire model (both hands): fingers along local -y, palm normal along local -x.
"""

import math
import threading

import mujoco
import numpy as np

from robot_model import ARM_REST, HAND_BODY_PREFIX, MODEL, STAND_LEGS, UPPER, lowest_point, set_hand_qpos

CENTER_DEG = 5.0  # |theta| below this: CENTER_ARM points
CENTER_ARM = "right"
MAX_AZIMUTH_DEG = 110.0  # beyond this the booth is behind the robot: not pointable
# elevation of the pointing ray vs horizontal distance (aim point at the 1 m reference height); log-distance interpolation
ELEV_CURVE = [(1.0, -45.0), (1.5, -40.0), (3.0, -25.0), (10.0, -10.0), (40.0, -3.0), (60.0, -2.0)]
REF_HEIGHT = 1.0
PALM_UP_DEG = 50.0  # 0 = palm faces inward (handshake), 90 = palm faces up
ELBOW_NOMINAL = 1.0  # rad (0 = right angle, ~1.45 = straight): about 25 deg bent
LIMIT_MARGIN = 0.08  # rad: IK keeps this far from the joint limits (the arm channel clips at 0.05)
ROLL_INWARD_MAX = -0.05  # shoulder roll toward the body (left arm sign); further in, the arm touches the torso
OUTWARD_MIN_DEG = 3.0  # the pointing ray is at least this far outward of the torso's facing direction
ELBOW_MAX = 1.35
MIN_ALONG = 0.30  # m: the wrist at least this far out along the ray from the shoulder  # the joint goes to 2.09, but past ~1.45 the elbow bends backwards (hyperextension): not natural
AZ_TOL_DEG, EL_TOL_DEG = 5.0, 6.0
DEBUG = False


def elev_curve(d):
    xs = [math.log(p[0]) for p in ELEV_CURVE]
    return float(np.interp(math.log(max(d, 0.3)), xs, [p[1] for p in ELEV_CURVE]))


def _unit(v):
    return v / np.linalg.norm(v)


class Pointer:
    """IK model of the robot standing (pelvis fixed, feet on the floor). Thread-safe (one solve at a time)."""

    def __init__(self):
        self.m = m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(m)
        self.lock = threading.Lock()
        self.qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in UPPER}
        self.dadr = {n: m.jnt_dofadr[m.joint(n).id] for n in UPPER}
        self.rng = {n: m.jnt_range[m.joint(n).id] for n in UPPER}
        self.rest = np.array([ARM_REST.get(n, 0.0) for n in UPPER])
        self.site = {s: (m.site(f"{s}_shoulder").id, m.site(f"{s}_palm").id) for s in ("left", "right")}
        # bodies of each arm (shoulder roll link downwards: the pitch link is the torso's child and always touches)
        self.arm_bodies = {}
        for s, pre in HAND_BODY_PREFIX.items():
            self.arm_bodies[s] = {b for b in range(m.nbody) if m.body(b).name.startswith((pre, f"{s}_shoulder_roll",
                                  f"{s}_shoulder_yaw", f"{s}_elbow", f"{s}_wrist"))}
        self._set_base()
        set_hand_qpos(m, self.d, "flat")  # IK and contact checks with flat hands (Dex3: thumb folded)
        q = self.rest.copy()
        self._apply(q)
        self.baseline = {s: self._arm_contacts(s) for s in ("left", "right")}  # contacts already present at rest
        self.reach = self._reach()

    # ---- model state
    def _set_base(self):
        m, d = self.m, self.d
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        for s in ("left", "right"):
            for j, v in STAND_LEGS.items():
                d.qpos[m.jnt_qposadr[m.joint(f"{s}_{j}_joint").id]] = v
        mujoco.mj_kinematics(m, d)
        d.qpos[2] = -lowest_point(m, d)

    def _apply(self, q15):
        for n, v in zip(UPPER, q15):
            self.d.qpos[self.qadr[n]] = v
        mujoco.mj_kinematics(self.m, self.d)

    def _reach(self):
        """Shoulder-to-wrist distance with the elbow at ELBOW_NOMINAL (arm otherwise straight)."""
        q = np.zeros(len(UPPER))
        q[UPPER.index("left_elbow_joint")] = ELBOW_NOMINAL
        self._apply(q)
        sh, pa = self.site["left"]
        r = float(np.linalg.norm(self.d.site_xpos[pa] - self.d.site_xpos[sh]))
        self._apply(self.rest)
        return r

    def _arm_contacts(self, side):
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        arm, out = self.arm_bodies[side], set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if c.dist < -0.002 and ((b1 in arm) != (b2 in arm)) and 0 not in (b1, b2):
                out.add(tuple(sorted((m.body(b1).name, m.body(b2).name))))
        return out

    # ---- geometry
    def geometry(self, aim, settings=None):
        st = settings or {}
        x, y, z = aim
        theta = math.degrees(math.atan2(y, x))
        d = math.hypot(x, y)
        k, pmax = st.get("torso_k", 0.25), st.get("torso_max_deg", 15.0)
        psi = max(-pmax, min(pmax, k * theta))
        side = "left" if theta > CENTER_DEG else "right" if theta < -CENTER_DEG else CENTER_ARM
        elev = elev_curve(d) + math.degrees(math.atan2(z - REF_HEIGHT, d))
        return {"azimuth_deg": theta, "distance_m": d, "torso_deg": psi, "torso_max_deg": pmax, "side": side, "elev_deg": elev,
                "pointable": abs(theta) <= MAX_AZIMUTH_DEG}

    # ---- IK
    def solve(self, aim, settings=None, q_seed=None):
        """Pointing pose for an aim point (robot frame). Returns geometry + q (15 upper joints, other arm at rest)
        + the achieved ray and the checks."""
        g = self.geometry(aim, settings)
        if not g["pointable"]:
            return {**g, "ok": False, "reason": "behind the robot (more than 110 deg to the side)"}
        with self.lock:
            res = self._solve(np.asarray(aim, float), g, q_seed)  # may adjust g["torso_deg"] (near, straight ahead)
            return {**g, **res}

    def _solve(self, aim, g, q_seed):
        m, d = self.m, self.d
        side = g["side"]
        names = [n for n in UPPER if n.startswith(side)]
        cols = [self.dadr[n] for n in names]
        lo = np.array([self.rng[n][0] for n in names]) + LIMIT_MARGIN
        hi = np.array([self.rng[n][1] for n in names]) - LIMIT_MARGIN
        hi[3] = min(hi[3], ELBOW_MAX)
        if side == "left":  # the upper arm never swings in toward the torso (roll 0 is already ~0.25 rad outward)
            lo[1] = max(lo[1], ROLL_INWARD_MAX)
        else:
            hi[1] = min(hi[1], -ROLL_INWARD_MAX)
        ix = [UPPER.index(n) for n in names]
        q = self.rest.copy() if q_seed is None else np.asarray(q_seed, float).copy()
        q[UPPER.index("waist_yaw_joint")] = math.radians(g["torso_deg"])
        other = "right" if side == "left" else "left"
        for n in UPPER:
            if n.startswith(other):
                q[UPPER.index(n)] = self.rest[UPPER.index(n)]
        sh_id, palm_id = self.site[side]
        wy = UPPER.index("waist_yaw_joint")
        pmax = math.radians(g.get("torso_max_deg", 15.0))
        for _ in range(3):
            # a near booth almost straight ahead lies slightly *inward* of the pointing shoulder; the arm never crosses
            # the body, so the torso turns a little further (toward the booth side of that shoulder) instead
            self._apply(q)
            S = d.site_xpos[sh_id].copy()
            az = math.atan2(aim[1] - S[1], aim[0] - S[0])
            rel = az - q[wy]
            need = math.radians(OUTWARD_MIN_DEG)
            if side == "right" and rel > -need:
                q[wy] = float(np.clip(az + need, -pmax, pmax))
            elif side == "left" and rel < need:
                q[wy] = float(np.clip(az - need, -pmax, pmax))
            else:
                break
        g["torso_deg"] = math.degrees(q[wy])
        el = math.radians(g["elev_deg"])
        u = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        # palm normal: inward (toward the body's midline, horizontal, perpendicular to the ray), turned up
        h = np.array([math.cos(az), math.sin(az), 0.0])
        inward = np.array([h[1], -h[0], 0.0]) if side == "left" else np.array([-h[1], h[0], 0.0])
        up = _unit(np.array([0, 0, 1.0]) - u[2] * u)
        inward = _unit(inward - (inward @ u) * u)
        b = math.radians(PALM_UP_DEG)
        n = _unit(math.cos(b) * inward + math.sin(b) * up)
        R_goal = np.column_stack([-n, -u, np.cross(-n, -u)])
        # wrist on the ray; its distance along the ray is free (the elbow stays near ELBOW_NOMINAL through the
        # null space), only kept at least MIN_ALONG out from the shoulder
        Pp = np.eye(3) - np.outer(u, u)
        q_nom = np.array([{"elbow": ELBOW_NOMINAL}.get(nm.split("_")[1], 0.0) for nm in names])
        W = np.array([1.0, 1.0, 1.0, 0.25, 0.25, 0.25])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        q_goal = np.zeros(4)
        mujoco.mju_mat2Quat(q_goal, R_goal.flatten())
        qcur, qdif = np.zeros(4), np.zeros(3)

        def error():
            mujoco.mju_mat2Quat(qcur, d.site_xmat[palm_id])
            mujoco.mju_subQuat(qdif, q_goal, qcur)  # rotation (local frame of the current palm) taking it to the goal
            rot = d.site_xmat[palm_id].reshape(3, 3) @ qdif
            v = d.site_xpos[palm_id] - S
            along = float(v @ u)
            return np.concatenate([-Pp @ v + u * max(0.0, MIN_ALONG - along), rot]), along

        def ik(qa, wr):
            """Stage 1 (first 60 iterations): wrist onto the ray only. Stage 2: the palm orientation (weight wr)
            blends in over 60 iterations. Going straight for the full 6D goal gets stuck at joint limits."""
            for it in range(300):
                q[ix] = qa
                self._apply(q)
                e, along = error()
                W[3:] = wr * min(1.0, max(0.0, (it - 60) / 60.0))
                if it > 120 and np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 3e-3:
                    break
                mujoco.mj_jacSite(m, d, jp, jr, palm_id)
                jpos = jp[:, cols] if along < MIN_ALONG else Pp @ jp[:, cols]
                J = np.vstack([jpos, jr[:, cols]]) * W[:, None]
                lam2 = 0.02 ** 2 + 0.1 * float(np.linalg.norm(W * e)) ** 2  # more damping far from the goal
                Jp = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(6))
                dq = Jp @ (W * e) + (np.eye(len(names)) - Jp @ J) @ (0.05 * (q_nom - qa))
                qa = np.clip(qa + np.clip(dq, -0.15, 0.15), lo, hi)
            q[ix] = qa
            self._apply(q)
            e, _ = error()
            return qa, float(np.linalg.norm(e[:3]) + 0.05 * np.linalg.norm(e[3:])), it + 1

        # seeds: upper arm roughly along the ray (shoulder pitch/roll from the direction; the model's shoulder has
        # a built-in ~0.25 rad roll outward), elbow nominal, a few elbow swivels (shoulder yaw)
        sgn = 1.0 if side == "left" else -1.0
        p0 = math.atan2(-u[0], -u[2])
        r0 = math.asin(float(np.clip(u[1], -1, 1))) - sgn * 0.25
        best = None
        seeds = [(0.25, p0, y) for y in (0.0, -0.5, 0.5, 1.0, -1.0, 1.5)]
        if abs(u[1]) > 0.7:  # arm out to the side: shoulder pitch is nearly free there (gimbal), try a few
            seeds += [(0.25, p, y) for p in (0.0, -0.6, 0.6) for y in (0.0, 1.0, -1.0)]
        seeds += [(0.03, p, y) for p, y in ((p0, 0.0), (p0, 0.5), (p0, 1.0), (p0, -0.5), (0.0, 0.0))]
        for wr, p_seed, yaw in seeds:
            if wr < 0.25 and best[1] < 0.003:
                break  # palm orientation only gives way (wr: its weight) when the full pose is not reachable
            seed = np.clip(np.array([p_seed, r0, yaw * sgn, ELBOW_NOMINAL, 0.0, 0.0, 0.0]), lo, hi)
            qa, cost, iters = ik(seed, wr)
            if DEBUG: print("seed", wr, round(p_seed, 2), yaw, "cost", round(cost, 4), "q", np.round(qa, 2))
            coll = self._arm_contacts(side) - self.baseline[side]
            near_lim = int(np.sum((qa - lo < 0.1) | (hi - qa < 0.1)))
            # prefer natural postures: elbow/wrist near nominal, away from the joint limits
            cost += (1.0 if coll else 0.0) + 0.003 * float(np.sum((qa - q_nom)[2:] ** 2)) + 0.01 * near_lim
            if best is None or cost < best[1]:
                best = (qa, cost, iters)
        qa, _, it = best
        it -= 1
        q[ix] = qa
        self._apply(q)
        P = d.site_xpos[palm_id].copy()
        S = d.site_xpos[sh_id].copy()
        ray = P - S
        ray_az = math.degrees(math.atan2(ray[1], ray[0]))
        ray_el = math.degrees(math.atan2(ray[2], math.hypot(ray[0], ray[1])))
        tgt_az = math.degrees(az)
        R = d.site_xmat[palm_id].reshape(3, 3)
        finger = -R[:, 1]
        az_err = (ray_az - tgt_az + 180) % 360 - 180
        el_err = ray_el - g["elev_deg"]
        coll = sorted(self._arm_contacts(side) - self.baseline[side])
        at_limit = [nm for nm, v, l, h_ in zip(names, qa, lo, hi) if v <= l + 1e-6 or v >= h_ - 1e-6]
        ok = abs(az_err) <= AZ_TOL_DEG and abs(el_err) <= EL_TOL_DEG and not coll
        reason = "" if ok else "; ".join(filter(None, [
            f"azimuth error {az_err:.1f} deg" if abs(az_err) > AZ_TOL_DEG else "",
            f"elevation error {el_err:.1f} deg" if abs(el_err) > EL_TOL_DEG else "",
            f"collision {coll}" if coll else ""]))
        self._apply(self.rest)
        return {"ok": ok, "reason": reason, "q": q.tolist(), "iterations": it + 1,
                "ray_azimuth_deg": ray_az, "ray_elev_deg": ray_el, "target_azimuth_deg": tgt_az,
                "azimuth_error_deg": az_err, "elev_error_deg": el_err,
                "finger_error_deg": math.degrees(math.acos(float(np.clip(finger @ u, -1, 1)))),
                "near_limit": [nm for nm, v, l, h_ in zip(names, qa, lo, hi) if v - l < 0.1 or h_ - v < 0.1],
                "wrist_off_ray_m": float(np.linalg.norm(Pp @ (P - S))), "at_limit": at_limit, "collisions": coll}

    def measure(self, q15, side):
        """Pointing ray (shoulder -> palm) of a measured pose: azimuth, elevation (deg)."""
        with self.lock:
            self._apply(np.asarray(q15, float))
            sh, pa = self.site[side]
            ray = self.d.site_xpos[pa] - self.d.site_xpos[sh]
            self._apply(self.rest)
        return math.degrees(math.atan2(ray[1], ray[0])), math.degrees(math.atan2(ray[2], math.hypot(ray[0], ray[1])))


def hand_frame(finger, palm_normal):
    """Rotation of the Inspire hand frame (fingers along local -y, palm normal along local -x) from world vectors."""
    u = _unit(np.asarray(finger, float))
    n = np.asarray(palm_normal, float)
    n = _unit(n - (n @ u) * u)
    return np.column_stack([-n, -u, np.cross(-n, -u)])


def hand_error(ptr, side, q15, p_goal, R_goal):
    """(position error m, rotation error deg, new contacts of this arm) of the palm site in pose q15."""
    m, d = ptr.m, ptr.d
    qg, qc, qd = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(qg, np.asarray(R_goal, float).flatten())
    with ptr.lock:
        ptr._apply(np.asarray(q15, float))
        palm = ptr.site[side][1]
        mujoco.mju_mat2Quat(qc, d.site_xmat[palm])
        mujoco.mju_subQuat(qd, qg, qc)
        pe, re_ = float(np.linalg.norm(p_goal - d.site_xpos[palm])), float(np.degrees(np.linalg.norm(qd)))
        coll = ptr._arm_contacts(side) - ptr.baseline[side]
        ptr._apply(ptr.rest)
    return pe, re_, coll


def solve_hand(ptr, side, p_goal, R_goal, q_base, w_rot=0.5, elbow_nom=0.6, seed_base=False, ignore_other_arm=False):
    """Full 6D IK of one arm: wrist (palm site) at p_goal with the hand frame R_goal; the waist and the other arm
    stay as in q_base. Several seeds, natural postures preferred; seed_base=True tries this arm's joints in q_base
    first and keeps that solution unless another seed is clearly better (used for mirrored, symmetric poses);
    ignore_other_arm=True does not count contacts with the other arm (the wai: its caller keeps the hands apart).
    Returns (q15, position error m, rotation error deg)."""
    m, d = ptr.m, ptr.d
    names = [n for n in UPPER if n.startswith(side)]
    cols = [ptr.dadr[n] for n in names]
    ix = [UPPER.index(n) for n in names]
    lo = np.array([ptr.rng[n][0] for n in names]) + LIMIT_MARGIN
    hi = np.array([ptr.rng[n][1] for n in names]) - LIMIT_MARGIN
    hi[3] = min(hi[3], ELBOW_MAX)
    if side == "left":
        lo[1] = max(lo[1], ROLL_INWARD_MAX)
    else:
        hi[1] = min(hi[1], -ROLL_INWARD_MAX)
    palm = ptr.site[side][1]
    qg = np.zeros(4)
    mujoco.mju_mat2Quat(qg, np.asarray(R_goal, float).flatten())
    qc, qd = np.zeros(4), np.zeros(3)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    q_nom = np.array([{"elbow": elbow_nom}.get(nm.split("_")[1], 0.0) for nm in names])
    W = np.ones(6)
    sgn = 1.0 if side == "left" else -1.0
    best = None
    with ptr.lock:
        q = np.asarray(q_base, float).copy()
        seeds = [np.array([p_s, sgn * r_s, sgn * y_s, elbow_nom, 0.0, 0.0, 0.0]) for p_s, r_s, y_s in
                 [(-0.5, 0.2, 0.0), (-1.0, 0.1, 0.5), (-0.3, 0.4, -0.5), (-1.2, 0.3, 1.0), (0.0, 0.2, 0.0)]]
        if seed_base:
            seeds.insert(0, q[ix].copy())
        for k_seed, seed in enumerate(seeds):
            qa = np.clip(seed, lo, hi)
            for it in range(300):
                q[ix] = qa
                ptr._apply(q)
                mujoco.mju_mat2Quat(qc, d.site_xmat[palm])
                mujoco.mju_subQuat(qd, qg, qc)
                e = np.concatenate([p_goal - d.site_xpos[palm], d.site_xmat[palm].reshape(3, 3) @ qd])
                W[3:] = w_rot * min(1.0, max(0.0, (it - 60) / 60.0))
                if it > 120 and np.linalg.norm(e[:3]) < 3e-4 and np.linalg.norm(e[3:]) < 5e-3:
                    break
                mujoco.mj_jacSite(m, d, jp, jr, palm)
                J = np.vstack([jp[:, cols], jr[:, cols]]) * W[:, None]
                lam2 = 0.02 ** 2 + 0.1 * float(np.linalg.norm(W * e)) ** 2
                Jp = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(6))
                dq = Jp @ (W * e) + (np.eye(7) - Jp @ J) @ (0.05 * (q_nom - qa))
                qa = np.clip(qa + np.clip(dq, -0.15, 0.15), lo, hi)
            q[ix] = qa
            ptr._apply(q)
            mujoco.mju_mat2Quat(qc, d.site_xmat[palm])
            mujoco.mju_subQuat(qd, qg, qc)
            pe, re_ = float(np.linalg.norm(p_goal - d.site_xpos[palm])), float(np.degrees(np.linalg.norm(qd)))
            coll = ptr._arm_contacts(side) - ptr.baseline[side]
            if ignore_other_arm:
                other = {m.body(b).name for b in ptr.arm_bodies["right" if side == "left" else "left"]}
                coll = {c for c in coll if not (set(c) & other)}
            near = int(np.sum((qa - lo < 0.1) | (hi - qa < 0.1)))
            cost = pe + 0.002 * re_ + (1.0 if coll else 0.0) + 0.01 * near - (0.02 if seed_base and k_seed == 0 else 0.0)
            if best is None or cost < best[0]:
                best = (cost, qa.copy(), pe, re_)
        q[ix] = best[1]
        ptr._apply(ptr.rest)
    return q, best[2], best[3]
