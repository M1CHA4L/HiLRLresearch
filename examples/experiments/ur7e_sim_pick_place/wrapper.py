import gymnasium as gym


class GripperPenaltyWrapper(gym.Wrapper):
    """Same as experiments/usb_pickup_insertion/wrapper.py (writes info["grasp_penalty"]),
    copied here so the sim config does not import RealSense camera code."""

    def __init__(self, env, penalty=-0.02):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]  # gripper_pose is first after flattening
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info
