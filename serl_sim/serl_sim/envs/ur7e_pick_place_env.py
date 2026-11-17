"""MuJoCo UR7e pick-and-place env with the same gym interface as franka_env.FrankaEnv.

Interface parity with the real-robot env (so all HIL-SERL wrappers work unchanged):
  * action: 7-dim in [-1, 1]  (dx, dy, dz, drx, dry, drz, gripper), base frame,
    target = current pose + action * ACTION_SCALE, then clipped to the safety box.
  * obs["state"]: tcp_pose (xyz + quat xyzw), tcp_vel (6, base frame), gripper_pose (1, 1=open),
    tcp_force (3), tcp_torque (3);  obs["images"]: {cam_name: (H, W, 3) uint8 RGB}
  * Cartesian impedance control with per-axis error clipping (same keys as COMPLIANCE_PARAM).
  * binary gripper with GRIPPER_SLEEP blocking, same thresholds as FrankaEnv.

Sim-only extras (never used as policy input):
  * get_privileged_state() for the simulated human / scripted expert.
  * info["failure"], info["failure_type"], info["contact_violation"] -> automatic failure
    labels for attention / risk-prediction research.
"""
import os
import time
import copy
from typing import Dict, Optional

import numpy as np
import gymnasium as gym
import mujoco
from scipy.spatial.transform import Rotation

_ASSET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "ur7e_pick_place.xml"
)

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
# elbow-up, TCP pointing down in front of the base
HOME_QPOS = np.array([np.pi, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
FINGER_OPEN = 0.04  # per-finger stroke [m]
GRIPPER_BODIES = ("gripper_base", "left_finger", "right_finger")


class SimEnvConfig:
    """Mirrors franka_env.envs.franka_env.DefaultEnvConfig, plus sim-only fields."""

    # ---------- same semantics as DefaultEnvConfig ----------
    RESET_POSE = np.array([0.45, 0.0, 0.15, np.pi, 0.0, 0.0])  # xyz + euler(xyz)
    ACTION_SCALE = np.array([0.015, 0.1, 1.0])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.03
    RANDOM_RZ_RANGE = 0.1
    ABS_POSE_LIMIT_LOW = np.array([0.30, -0.22, 0.005, np.pi - 0.15, -0.15, -0.8])
    ABS_POSE_LIMIT_HIGH = np.array([0.62, 0.25, 0.25, np.pi, 0.15, 0.8])
    COMPLIANCE_PARAM: Dict[str, float] = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
    }
    GRIPPER_SLEEP = 0.6
    MAX_EPISODE_LENGTH = 120
    DISPLAY_IMAGE = False

    # ---------- sim-only ----------
    IMAGE_KEYS = ("front", "wrist")
    IMAGE_SIZE = 128
    CUBE_XY_LOW = np.array([0.40, -0.15])
    CUBE_XY_HIGH = np.array([0.52, -0.05])
    CUBE_YAW_RANGE = 0.4  # rad, uniform in [-r, r]
    PLACE_XY = np.array([0.46, 0.15])
    PLACE_XY_RANDOM = 0.0  # +- range around PLACE_XY
    TARGET_RADIUS = 0.03  # success if cube center within this xy distance of the target
    TABLE_X = (0.15, 0.85)
    TABLE_Y = (-0.45, 0.45)
    FORCE_LIMIT = 30.0  # [N] external force above this flags info["contact_violation"]
    REALTIME = False  # True: sleep so that env.step runs at `hz` like the real robot
    LAUNCH_VIEWER = False  # interactive mujoco viewer (needs a display)


class UR7ePickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        hz: int = 10,
        fake_env: bool = False,
        save_video: bool = False,
        config: Optional[SimEnvConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or SimEnvConfig()
        c = self.config
        self.hz = hz
        self.fake_env = fake_env
        self.save_video = save_video
        self.action_scale = c.ACTION_SCALE
        self.max_episode_length = c.MAX_EPISODE_LENGTH
        self._rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_path(_ASSET)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        m = self.model

        self._arm_qpos = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self._arm_dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self._lf_qpos = m.jnt_qposadr[m.joint("left_finger_joint").id]
        self._rf_qpos = m.jnt_qposadr[m.joint("right_finger_joint").id]
        self._cube_qpos = m.jnt_qposadr[m.joint("cube_joint").id]
        self._cube_dof = m.jnt_dofadr[m.joint("cube_joint").id]
        self._tcp = m.site("tcp").id
        self._target_body = m.body("target").id
        self._cube_geom = m.geom("cube").id
        self._lpad = m.geom("left_pad").id
        self._rpad = m.geom("right_pad").id
        self._gripper_bodies = {m.body(b).id for b in GRIPPER_BODIES}
        self._gripper_act = m.actuator("gripper").id
        self._n_substeps = max(1, int(round(1.0 / (hz * m.opt.timestep))))
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self._set_ctrl_param(c.COMPLIANCE_PARAM)

        self.xyz_bounding_box = gym.spaces.Box(
            c.ABS_POSE_LIMIT_LOW[:3], c.ABS_POSE_LIMIT_HIGH[:3], dtype=np.float64
        )
        self.rpy_bounding_box = gym.spaces.Box(
            c.ABS_POSE_LIMIT_LOW[3:], c.ABS_POSE_LIMIT_HIGH[3:], dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1, np.ones((7,), dtype=np.float32)
        )
        s = c.IMAGE_SIZE
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        k: gym.spaces.Box(0, 255, shape=(s, s, 3), dtype=np.uint8)
                        for k in c.IMAGE_KEYS
                    }
                ),
            }
        )

        self._renderer = None
        self._viewer = None
        self.recording_frames = []
        self.curr_path_length = 0
        self.terminate = False
        self._target = np.zeros(7)
        self._grip_cmd_open = True
        self._last_grip_time = -np.inf
        self._episode_max_force = 0.0

        # initialise to a valid state so observation queries work before reset()
        self._reset_sim()

    # ------------------------------------------------------------------ control
    def _set_ctrl_param(self, p: Dict[str, float]):
        self._kt = float(p["translational_stiffness"])
        self._dt = float(p["translational_damping"])
        self._kr = float(p["rotational_stiffness"])
        self._dr = float(p["rotational_damping"])
        self._clip_t_hi = np.array([p[f"translational_clip_{a}"] for a in "xyz"])
        self._clip_t_lo = -np.array([p[f"translational_clip_neg_{a}"] for a in "xyz"])
        self._clip_r_hi = np.array([p[f"rotational_clip_{a}"] for a in "xyz"])
        self._clip_r_lo = -np.array([p[f"rotational_clip_neg_{a}"] for a in "xyz"])

    def _tcp_jacobian(self, data):
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self._tcp)
        return np.vstack([self._jacp[:, self._arm_dof], self._jacr[:, self._arm_dof]])

    def _site_quat_wxyz(self, data):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, data.site_xmat[self._tcp])
        return q

    def _impedance_torque(self):
        """tau = J^T [K e - D v] + bias, with Franka-style clipping of the pose error."""
        d = self.data
        J = self._tcp_jacobian(d)
        v = J @ d.qvel[self._arm_dof]
        p = d.site_xpos[self._tcp]
        e_p = np.clip(self._target[:3] - p, self._clip_t_lo, self._clip_t_hi)

        q_cur = self._site_quat_wxyz(d)
        q_tgt = np.array([self._target[6], *self._target[3:6]])  # xyzw -> wxyz
        e_local = np.zeros(3)
        mujoco.mju_subQuat(e_local, q_tgt, q_cur)  # q_cur * exp(e_local) = q_tgt
        e_r = d.site_xmat[self._tcp].reshape(3, 3) @ e_local  # to base frame
        e_r = np.clip(e_r, self._clip_r_lo, self._clip_r_hi)

        F = np.concatenate([self._kt * e_p - self._dt * v[:3], self._kr * e_r - self._dr * v[3:]])
        return J.T @ F + d.qfrc_bias[self._arm_dof]

    def _simulate(self, duration: float):
        n = max(1, int(round(duration / self.model.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step1(self.model, self.data)
            self.data.ctrl[:6] = self._impedance_torque()
            mujoco.mj_step2(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        f = np.linalg.norm(self._external_wrench()[0])
        self._episode_max_force = max(self._episode_max_force, f)

    # ------------------------------------------------------------- FrankaEnv API
    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Identical to FrankaEnv.clip_safety_box."""
        pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(np.abs(euler[0]), self.rpy_bounding_box.low[0], self.rpy_bounding_box.high[0])
        )
        euler[1:] = np.clip(euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:])
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()
        return pose

    def step(self, action: np.ndarray):
        start = time.time()
        action = np.clip(np.asarray(action, dtype=np.float64), self.action_space.low, self.action_space.high)
        self._update_currpos()

        nextpos = self.currpos.copy()
        nextpos[:3] = nextpos[:3] + action[:3] * self.action_scale[0]
        nextpos[3:] = (
            Rotation.from_rotvec(action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        self._send_gripper_command(action[6] * self.action_scale[2])
        self._target = self.clip_safety_box(nextpos)
        self._simulate(1.0 / self.hz)
        self.curr_path_length += 1

        self._update_currpos()
        ob = self._get_obs()
        succeed = self._is_success()
        failure_type = self._failure_type()
        timeout = self.curr_path_length >= self.max_episode_length
        done = bool(timeout or succeed or failure_type is not None or self.terminate)
        f_ext, _ = self._external_wrench()
        info = {
            "succeed": bool(succeed),
            "failure": failure_type is not None,
            "failure_type": failure_type or "",
            "timeout": bool(timeout and not succeed),
            "contact_violation": bool(np.linalg.norm(f_ext) > self.config.FORCE_LIMIT),
            "grasped": self._is_grasped(),
        }
        if self.config.REALTIME:
            time.sleep(max(0.0, 1.0 / self.hz - (time.time() - start)))
        return ob, int(succeed), done, False, info

    def compute_reward(self, obs=None) -> bool:
        return self._is_success()

    def reset(self, seed: Optional[int] = None, options=None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.save_video:
            self.save_video_recording()
        self._reset_sim()
        self.curr_path_length = 0
        self.terminate = False
        self._episode_max_force = 0.0
        self._update_currpos()
        return self._get_obs(), {"succeed": False}

    # -------------------------------------------------------------------- reset
    def _reset_sim(self):
        c = self.config
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)

        pose = c.RESET_POSE.copy()
        if c.RANDOM_RESET:
            pose[:2] += self._rng.uniform(-c.RANDOM_XY_RANGE, c.RANDOM_XY_RANGE, 2)
            pose[5] += self._rng.uniform(-c.RANDOM_RZ_RANGE, c.RANDOM_RZ_RANGE)
        quat = Rotation.from_euler("xyz", pose[3:]).as_quat()
        d.qpos[self._arm_qpos] = self._ik(pose[:3], quat)
        d.qpos[self._lf_qpos] = FINGER_OPEN
        d.qpos[self._rf_qpos] = FINGER_OPEN
        d.ctrl[self._gripper_act] = FINGER_OPEN
        self._grip_cmd_open = True
        self._last_grip_time = -np.inf

        place = c.PLACE_XY + self._rng.uniform(-c.PLACE_XY_RANDOM, c.PLACE_XY_RANDOM, 2)
        m.body_pos[self._target_body][:2] = place
        cube_xy = self._rng.uniform(c.CUBE_XY_LOW, c.CUBE_XY_HIGH)
        yaw = self._rng.uniform(-c.CUBE_YAW_RANGE, c.CUBE_YAW_RANGE)
        q_wxyz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        d.qpos[self._cube_qpos:self._cube_qpos + 7] = np.concatenate([cube_xy, [0.0205], q_wxyz])
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

        self._target = np.concatenate([pose[:3], quat])
        self._simulate(0.3)  # settle

    def _ik(self, pos, quat_xyzw, iters: int = 300):
        """Damped least-squares IK on a scratch MjData; seeded from HOME_QPOS."""
        m, d = self.model, self._ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self._arm_qpos] = HOME_QPOS
        q_tgt = np.array([quat_xyzw[3], *quat_xyzw[:3]])
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e_p = pos - d.site_xpos[self._tcp]
            q_cur = self._site_quat_wxyz(d)
            e_l = np.zeros(3)
            mujoco.mju_subQuat(e_l, q_tgt, q_cur)
            err = np.concatenate([e_p, d.site_xmat[self._tcp].reshape(3, 3) @ e_l])
            if np.linalg.norm(err) < 1e-5:
                break
            J = self._tcp_jacobian(d)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d.qpos[self._arm_qpos] += np.clip(dq, -0.2, 0.2)
        return d.qpos[self._arm_qpos].copy()

    # ------------------------------------------------------------------ gripper
    def _send_gripper_command(self, pos: float, mode: str = "binary"):
        """Same thresholds as FrankaEnv; arm holds its target while the gripper moves."""
        if mode != "binary":
            raise NotImplementedError
        now = self.data.time
        g = float(self.curr_gripper_pos[0])
        if pos <= -0.5 and g > 0.85 and now - self._last_grip_time > self.config.GRIPPER_SLEEP:
            self.data.ctrl[self._gripper_act] = 0.0
            self._grip_cmd_open = False
        elif pos >= 0.5 and g < 0.85 and now - self._last_grip_time > self.config.GRIPPER_SLEEP:
            self.data.ctrl[self._gripper_act] = FINGER_OPEN
            self._grip_cmd_open = True
        else:
            return
        self._last_grip_time = now
        self._simulate(self.config.GRIPPER_SLEEP)

    # ------------------------------------------------------------- state / obs
    def _external_wrench(self):
        """Contact force/torque applied by the environment on the gripper (base frame, about TCP)."""
        m, d = self.model, self.data
        f_tot, t_tot = np.zeros(3), np.zeros(3)
        p_tcp = d.site_xpos[self._tcp]
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            b1 = m.geom_bodyid[con.geom1] in self._gripper_bodies
            b2 = m.geom_bodyid[con.geom2] in self._gripper_bodies
            if b1 == b2:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            # contact frame rows: normal (geom1 -> geom2), tangent1, tangent2.
            # f6 is the force geom1 exerts on geom2 in that frame.
            f_world = con.frame.reshape(3, 3).T @ f6[:3]
            f_on_gripper = f_world if b2 else -f_world
            f_tot += f_on_gripper
            t_tot += np.cross(con.pos - p_tcp, f_on_gripper)
        return f_tot, t_tot

    def _update_currpos(self):
        d = self.data
        q = self._site_quat_wxyz(d)
        self.currpos = np.concatenate([d.site_xpos[self._tcp], [q[1], q[2], q[3], q[0]]])
        J = self._tcp_jacobian(d)
        self.currvel = J @ d.qvel[self._arm_dof]
        self.currforce, self.currtorque = self._external_wrench()
        self.q = d.qpos[self._arm_qpos].copy()
        self.dq = d.qvel[self._arm_dof].copy()
        self.currjacobian = J
        self.curr_gripper_pos = np.array(
            [np.clip(d.qpos[self._lf_qpos] / FINGER_OPEN, 0.0, 1.0)]
        )

    def _render_images(self) -> Dict[str, np.ndarray]:
        s = self.config.IMAGE_SIZE
        if self.fake_env:
            return {k: np.zeros((s, s, 3), np.uint8) for k in self.config.IMAGE_KEYS}
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, s, s)
        imgs = {}
        for k in self.config.IMAGE_KEYS:
            self._renderer.update_scene(self.data, camera=k)
            imgs[k] = self._renderer.render().copy()
        return imgs

    def _get_obs(self) -> dict:
        images = self._render_images()
        if self.save_video:
            self.recording_frames.append({k: v[..., ::-1] for k, v in images.items()})
        if self.config.DISPLAY_IMAGE and not self.fake_env:
            import cv2

            cv2.imshow("ur7e_sim", np.concatenate([v[..., ::-1] for v in images.values()], axis=1))
            cv2.waitKey(1)
        if self.config.LAUNCH_VIEWER and not self.fake_env:
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        state = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state))

    # --------------------------------------------------------- task predicates
    def _cube_pose(self):
        q = self.data.qpos[self._cube_qpos:self._cube_qpos + 7]
        return q[:3].copy(), np.array([q[4], q[5], q[6], q[3]])  # pos, quat xyzw

    def _contact_between(self, g1, g2) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == g1 and c.geom2 == g2) or (c.geom1 == g2 and c.geom2 == g1):
                return True
        return False

    def _is_grasped(self) -> bool:
        return self._contact_between(self._lpad, self._cube_geom) and self._contact_between(
            self._rpad, self._cube_geom
        )

    def _place_xy(self):
        return self.model.body_pos[self._target_body][:2].copy()

    def _is_success(self) -> bool:
        pos, _ = self._cube_pose()
        in_target = np.linalg.norm(pos[:2] - self._place_xy()) < self.config.TARGET_RADIUS
        on_table = pos[2] < 0.03
        return bool(in_target and on_table and not self._is_grasped())

    def _failure_type(self) -> Optional[str]:
        pos, _ = self._cube_pose()
        c = self.config
        if pos[2] < -0.03 or not (c.TABLE_X[0] < pos[0] < c.TABLE_X[1] and c.TABLE_Y[0] < pos[1] < c.TABLE_Y[1]):
            return "object_off_table"
        return None

    @staticmethod
    def _cube_face_yaw(quat_xyzw) -> float:
        """Yaw of the most horizontal cube axis (robust when the cube is tipped over)."""
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        ax = R[:, int(np.argmin(np.abs(R[2, :])))]
        return float(np.arctan2(ax[1], ax[0]))

    def get_privileged_state(self) -> dict:
        """Ground-truth state for the simulated human / analysis. Not a policy input."""
        cube_pos, cube_quat = self._cube_pose()
        return {
            "tcp_pos": self.currpos[:3].copy(),
            "tcp_quat": self.currpos[3:].copy(),
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
            "cube_yaw": self._cube_face_yaw(cube_quat),
            "place_xy": self._place_xy(),
            "gripper_open": float(self.curr_gripper_pos[0]),
            "gripper_cmd_open": self._grip_cmd_open,
            "grasped": self._is_grasped(),
            "success": self._is_success(),
            "episode_max_force": self._episode_max_force,
        }

    # --------------------------------------------------------------- utilities
    def save_video_recording(self):
        if not self.recording_frames:
            return
        try:
            import cv2
            from datetime import datetime

            os.makedirs("./videos", exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for key in self.recording_frames[0]:
                h, w = self.recording_frames[0][key].shape[:2]
                vw = cv2.VideoWriter(f"./videos/sim_{key}_{ts}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), self.hz, (w, h))
                for fr in self.recording_frames:
                    vw.write(np.ascontiguousarray(fr[key]))
                vw.release()
        except Exception as e:  # noqa: BLE001
            print(f"Failed to save video: {e}")
        self.recording_frames.clear()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
