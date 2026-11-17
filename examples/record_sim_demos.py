#!/usr/bin/env python3
"""Collect demos in simulation with the scripted expert (same output format as record_demos.py).

    cd examples && python record_sim_demos.py --exp_name=ur7e_sim_pick_place --successes_needed=20
"""
import copy
import datetime
import os
import pickle as pkl

import numpy as np
from absl import app, flags
from tqdm import tqdm

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "ur7e_sim_pick_place", "Sim experiment name.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_integer("seed", 0, "Sim seed (use a different seed than training).")
flags.DEFINE_float("expert_noise", 0.05, "Std of Gaussian noise on expert actions.")
flags.DEFINE_string("out_dir", "./demo_data", "Output directory.")


def main(_):
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    config.sim_seed = FLAGS.seed
    config.realtime = False
    config.human_kwargs = dict(config.human_kwargs, noise_std=FLAGS.expert_noise)
    env = config.get_environment(fake_env=False, save_video=False, classifier=True, human_mode="expert")

    obs, _ = env.reset()
    transitions, trajectory = [], []
    successes, attempts = 0, 0
    pbar = tqdm(total=FLAGS.successes_needed)
    while successes < FLAGS.successes_needed:
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]
        info.pop("original_state_obs", None)
        trajectory.append(
            copy.deepcopy(
                dict(observations=obs, actions=actions, next_observations=next_obs, rewards=rew,
                     masks=1.0 - done, dones=done, infos=info)
            )
        )
        obs = next_obs
        if done or truncated:
            attempts += 1
            if info["succeed"]:
                transitions.extend(trajectory)
                successes += 1
                pbar.update(1)
            trajectory = []
            obs, _ = env.reset()

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(FLAGS.out_dir, f"{FLAGS.exp_name}_{FLAGS.successes_needed}_demos_{ts}.pkl")
    with open(path, "wb") as f:
        pkl.dump(transitions, f)
    print(f"saved {successes} demos ({len(transitions)} transitions, {successes}/{attempts} success) to {path}")


if __name__ == "__main__":
    app.run(main)
