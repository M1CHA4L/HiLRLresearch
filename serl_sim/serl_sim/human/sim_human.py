"""Simulated human operator for HIL-SERL in simulation.

The wrapper sits exactly where SpacemouseIntervention sits on the real robot (directly on the base
env, below RelativeFrame), receives base-frame actions, and reports takeovers through
info["intervene_action"] so train_rlpd.py / record_demos.py work unchanged.

Modes
-----
  "off"    : no human (pure RLPD + demos baseline).
  "expert" : the expert always acts (demo collection).
  "full"   : always-attentive human = standard HIL-SERL. Attention cost is 100% of steps.
  "alert"  : the human is away (doing a secondary task) until request_attention() is called.
             Then: reaction phase (switching cost, counts as attention, no control) ->
             attending phase (watches, may intervene using the "full" rule) -> away again.

Attention accounting (per step): info["attention"] in {0, 1}; per-episode totals are put in
info["attention_stats"] when the episode ends.
"""
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation


class ScriptedPickPlaceExpert:
    """State-machine expert using privileged state. Outputs a normalized base-frame action."""

    def __init__(self, action_scale, hover_z=0.10, noise_std=0.0, rng=None):
        self.action_scale = np.asarray(action_scale, dtype=np.float64)
        self.hover_z = hover_z
        self.noise_std = noise_std
        self.rng = rng or np.random.default_rng()
        self.phase = ""

    def act(self, s: dict) -> np.ndarray:
        p, c, place = s["tcp_pos"], s["cube_pos"], s["place_xy"]
        g_open, grasped = s["gripper_open"], s["grasped"]
        grip = 0.0
        yaw_des = None

        if grasped:
            xy_err = np.linalg.norm(p[:2] - place)
            place_z = 0.022
            if xy_err > 0.012:
                if p[2] < self.hover_z - 0.02 and xy_err > 0.03:
                    goal, self.phase = np.array([p[0], p[1], self.hover_z]), "lift"
                else:
                    goal, self.phase = np.array([place[0], place[1], self.hover_z]), "transport"
            else:
                goal, self.phase = np.array([place[0], place[1], place_z]), "lower"
                if p[2] - place_z < 0.012:
                    grip, self.phase = 1.0, "release"
        else:
            if s["success"]:
                goal, self.phase = p + np.array([0, 0, 0.03]), "done"
                grip = 1.0
            elif g_open < 0.85:  # closed on nothing -> reopen and back off
                goal, self.phase = p + np.array([0, 0, 0.02]), "reopen"
                grip = 1.0
            else:
                xy_err = np.linalg.norm(p[:2] - c[:2])
                # align finger axis with a cube face (mod 90 deg)
                yaw_des = (s["cube_yaw"] + np.pi / 4) % (np.pi / 2) - np.pi / 4
                if xy_err > 0.008:
                    if p[2] < c[2] + 0.04 and xy_err > 0.02:
                        goal, self.phase = np.array([p[0], p[1], self.hover_z]), "rise"
                    else:
                        goal, self.phase = np.array([c[0], c[1], max(self.hover_z, p[2] - 0.0)]), "approach"
                        goal[2] = self.hover_z if xy_err > 0.02 else p[2]
                else:
                    goal, self.phase = np.array([c[0], c[1], c[2]]), "descend"
                    if abs(p[2] - c[2]) < 0.008:
                        grip, self.phase = -1.0, "close"

        a = np.zeros(7)
        a[:3] = np.clip((goal - p) / self.action_scale[0], -1, 1)

        r_cur = Rotation.from_quat(s["tcp_quat"])
        if yaw_des is None:
            yaw_des = r_cur.as_euler("xyz")[2]  # keep current yaw, fix roll/pitch
        r_des = Rotation.from_euler("xyz", [np.pi, 0.0, yaw_des])
        a[3:6] = np.clip((r_des * r_cur.inv()).as_rotvec() / self.action_scale[1], -1, 1)
        a[6] = grip

        if self.noise_std > 0:
            a[:6] += self.rng.normal(0, self.noise_std, 6)
        a[:6] = np.clip(a[:6], -1, 1)
        return a


class SimHumanIntervention(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        mode: str = "full",
        expert: Optional[ScriptedPickPlaceExpert] = None,
        reaction_steps: Tuple[int, int] = (10, 30),  # 1-3 s at 10 Hz
        attend_steps: int = 30,  # how long the human watches after an alert
        patience: int = 3,  # consecutive "bad" policy steps before taking over
        hold_steps: int = 10,  # minimum length of one takeover
        max_hold_steps: int = 40,  # a takeover continues until the expert's sub-skill (phase) changes
        disagree_cos: float = 0.0,  # policy/expert translation cosine below this = "bad"
        noise_std: float = 0.05,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        assert mode in ("off", "expert", "full", "alert"), mode
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.expert = expert or ScriptedPickPlaceExpert(
            env.unwrapped.action_scale, noise_std=noise_std, rng=self.rng
        )
        self.reaction_steps = reaction_steps
        self.attend_steps = attend_steps
        self.patience = patience
        self.hold_steps = hold_steps
        self.max_hold_steps = max_hold_steps
        self.disagree_cos = disagree_cos

        # lifetime counters (for logging attention over training)
        self.total_steps = 0
        self.total_attention_steps = 0
        self.total_alerts = 0
        self._reset_episode_state()
        self._human_state = "away"  # away | reacting | attending  (persists across episodes)
        self._state_timer = 0
        self._alert_pending = False

    # ------------------------------------------------------------------ API
    def request_attention(self):
        """Called by the alert scheduler (risk module). Takes effect at the next step."""
        self._alert_pending = True

    @property
    def human_state(self) -> str:
        return self._human_state if self.mode == "alert" else ("attending" if self.mode != "off" else "away")

    # ------------------------------------------------------------ internals
    def _reset_episode_state(self):
        self._bad_count = 0
        self._hold = 0
        self._hold_elapsed = 0
        self._hold_phase = ""
        self.ep_attention = 0
        self.ep_interventions = 0
        self.ep_intervention_steps = 0
        self.ep_alerts = 0
        self.ep_steps = 0

    def _advance_alert_fsm(self) -> bool:
        """Returns whether the human is paying attention this step."""
        if self._alert_pending:
            self._alert_pending = False
            self.ep_alerts += 1
            self.total_alerts += 1
            if self._human_state == "away":
                self._human_state = "reacting"
                self._state_timer = int(self.rng.integers(self.reaction_steps[0], self.reaction_steps[1] + 1))
            elif self._human_state == "attending":
                self._state_timer = max(self._state_timer, self.attend_steps)  # extend
        if self._human_state == "reacting":
            self._state_timer -= 1
            if self._state_timer <= 0:
                self._human_state, self._state_timer = "attending", self.attend_steps
            return True
        if self._human_state == "attending":
            self._state_timer -= 1
            if self._state_timer <= 0 and not self._in_takeover():
                self._human_state = "away"
            return True
        return False

    def _in_takeover(self) -> bool:
        """Hold for >= hold_steps, then until the sub-skill being corrected is finished."""
        if self._hold_phase == "":
            return False
        if self._hold > 0 or (self.expert.phase == self._hold_phase and self._hold_elapsed < self.max_hold_steps):
            return True
        self._hold_phase = ""
        return False

    def _wants_takeover(self, policy_a: np.ndarray, expert_a: np.ndarray, s: dict) -> bool:
        """Rule of an attentive operator: take over when the policy keeps doing the wrong thing."""
        bad = False
        e, pa = expert_a[:3], policy_a[:3]
        if np.linalg.norm(e) > 0.3:
            cos = float(e @ pa) / (np.linalg.norm(e) * np.linalg.norm(pa) + 1e-8)
            stalled = np.linalg.norm(pa) < 0.1  # policy idles while progress is needed
            bad = cos < self.disagree_cos or stalled
        # wrong gripper decision at a critical moment
        if expert_a[6] != 0 and np.sign(policy_a[6]) == -np.sign(expert_a[6]) and abs(policy_a[6]) > 0.5:
            bad = True
        if s["grasped"] is False and self.expert.phase in ("transport", "lower"):
            bad = True  # dropped the object
        self._bad_count = self._bad_count + 1 if bad else 0
        return self._bad_count >= self.patience

    # ------------------------------------------------------------- gym API
    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        s = self.env.unwrapped.get_privileged_state()
        expert_a = self.expert.act(s)

        if self.mode == "off":
            attending = False
        elif self.mode in ("expert", "full"):
            attending = True
        else:
            attending = self._advance_alert_fsm()
        can_control = attending and (self.mode != "alert" or self._human_state == "attending")

        intervene = False
        if self.mode == "expert":
            intervene = True
        elif can_control:
            if self._in_takeover():
                intervene = True
            elif self._wants_takeover(action, expert_a, s):
                self._hold, self._hold_elapsed = self.hold_steps, 0
                self._hold_phase = self.expert.phase
                self.ep_interventions += 1
                self._bad_count = 0
                intervene = True
        if intervene and self.mode != "expert":
            self._hold = max(0, self._hold - 1)
            self._hold_elapsed += 1

        new_action = expert_a if intervene else action
        obs, rew, done, truncated, info = self.env.step(new_action)
        if intervene:
            info["intervene_action"] = new_action
            self.ep_intervention_steps += 1

        self.ep_steps += 1
        self.ep_attention += int(attending)
        self.total_steps += 1
        self.total_attention_steps += int(attending)
        info["attention"] = int(attending)
        info["human_state"] = self.human_state
        if done or truncated:
            info["attention_stats"] = {
                "attention_steps": self.ep_attention,
                "attention_frac": self.ep_attention / max(1, self.ep_steps),
                "alerts": self.ep_alerts,
                "takeovers": self.ep_interventions,
                "takeover_steps": self.ep_intervention_steps,
                "cum_attention_steps": self.total_attention_steps,
                "cum_attention_frac": self.total_attention_steps / max(1, self.total_steps),
                "failure": int(info.get("failure", False)),
            }
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_episode_state()
        return obs, info
