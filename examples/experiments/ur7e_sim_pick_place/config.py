"""HIL-SERL on a simulated UR7e pick-and-place task (MuJoCo), with a simulated human.

Modes are set in this file or overridden from train_rlpd.py flags:
  --human_mode       off | full | alert | expert | spacemouse | ros2
  --alert_risk       random | q_std | neg_q          (only for human_mode=alert)
  --attention_budget 0.2
  --ros2_bridge      publish state / images / attention on ROS2 topics
"""
import numpy as np

from franka_env.envs.wrappers import Quat2EulerWrapper
from franka_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from serl_sim.envs.ur7e_pick_place_env import UR7ePickPlaceEnv, SimEnvConfig
from serl_sim.human.sim_human import SimHumanIntervention

from experiments.config import DefaultTrainingConfig
from experiments.ur7e_sim_pick_place.wrapper import GripperPenaltyWrapper


class EnvConfig(SimEnvConfig):
    RESET_POSE = np.array([0.45, 0.0, 0.15, np.pi, 0.0, 0.0])
    ACTION_SCALE = np.array([0.015, 0.1, 1.0])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.03
    RANDOM_RZ_RANGE = 0.1
    CUBE_XY_LOW = np.array([0.40, -0.15])
    CUBE_XY_HIGH = np.array([0.52, -0.05])
    CUBE_YAW_RANGE = 0.4
    PLACE_XY = np.array([0.46, 0.15])
    MAX_EPISODE_LENGTH = 120
    IMAGE_KEYS = ("front", "wrist")
    IMAGE_SIZE = 128
    DISPLAY_IMAGE = False
    REALTIME = False  # True = 10 Hz wall clock like the real robot (needed with a real human / ROS2 teleop)


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front", "wrist"]
    classifier_keys = ["front"]  # unused in sim: reward is ground truth
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    buffer_period = 1000
    cta_ratio = 2
    random_steps = 0
    discount = 0.97
    max_traj_length = 120
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    # Critic ensemble. HIL-SERL default is 2, which is useless as an uncertainty estimate.
    # REDQ-style: 10 critics, min over a random subset of 2 for the target.
    critic_ensemble_size = 10
    critic_subsample_size = 2

    # Simulated human / attention budget
    human_mode = "full"
    alert_risk = "q_std"
    attention_budget = 0.2
    use_miss_controller = True
    human_kwargs = dict(reaction_steps=(10, 30), attend_steps=30, patience=3, hold_steps=10, max_hold_steps=40,
                        noise_std=0.05)

    ros2_bridge = False
    # True: the actor steps at 10 Hz wall clock like the real robot, so the learner's
    # update-to-data ratio matches real HIL-SERL. --norealtime runs as fast as possible
    # (much faster, but the UTD ratio then depends on GPU speed -> not comparable to the paper).
    realtime = True
    sim_seed = 0

    def get_environment(self, fake_env=False, save_video=False, classifier=False, human_mode=None):
        mode = human_mode or self.human_mode
        env_cfg = EnvConfig()
        env_cfg.REALTIME = self.realtime or mode in ("spacemouse", "ros2")
        env = UR7ePickPlaceEnv(fake_env=fake_env, save_video=save_video, config=env_cfg, seed=self.sim_seed)

        if not fake_env:
            if mode == "spacemouse":
                from franka_env.envs.wrappers import SpacemouseIntervention

                env = SpacemouseIntervention(env)
            elif mode == "ros2":
                from serl_sim.ros2.bridge import Ros2SpacenavIntervention

                env = Ros2SpacenavIntervention(env)
            else:
                env = SimHumanIntervention(env, mode=mode, seed=self.sim_seed, **self.human_kwargs)
            if self.ros2_bridge:
                from serl_sim.ros2.bridge import Ros2BridgeWrapper

                env = Ros2BridgeWrapper(env)

        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # `classifier` is ignored: the sim provides the ground-truth success reward.
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
