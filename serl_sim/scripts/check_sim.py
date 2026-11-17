"""Sanity check: run the simulated human in a given mode and report success / attention / speed.

    python scripts/check_sim.py --mode expert --episodes 20            # scripted expert should be ~100%
    python scripts/check_sim.py --mode full --policy random --episodes 20   # HIL with a random "policy"
    python scripts/check_sim.py --mode expert --save_frames frames.png
"""
import argparse
import time

import numpy as np

from serl_sim.envs.ur7e_pick_place_env import UR7ePickPlaceEnv, SimEnvConfig
from serl_sim.human.sim_human import SimHumanIntervention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="expert", choices=["off", "expert", "full", "alert"])
    ap.add_argument("--policy", default="zero", choices=["zero", "random"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    ap.add_argument("--save_frames", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SimEnvConfig()
    cfg.LAUNCH_VIEWER = args.viewer
    cfg.REALTIME = args.viewer
    env = SimHumanIntervention(
        UR7ePickPlaceEnv(fake_env=args.no_render, config=cfg, seed=args.seed), mode=args.mode, seed=args.seed
    )
    rng = np.random.default_rng(args.seed)
    succ, steps, att, takeovers, frames = 0, 0, 0, 0, []
    t0 = time.time()
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            a = np.zeros(7) if args.policy == "zero" else rng.uniform(-1, 1, 7)
            if args.mode == "alert" and rng.random() < 0.02:
                env.request_attention()
            obs, r, done, _, info = env.step(a)
            steps += 1
            if args.save_frames and ep == 0 and steps % 10 == 0 and not args.no_render:
                frames.append(np.concatenate(list(obs["images"].values()), axis=1))
        st = info["attention_stats"]
        succ += int(info["succeed"])
        att += st["attention_steps"]
        takeovers += st["takeovers"]
        print(f"ep {ep:3d} succeed={info['succeed']} len={st['attention_steps'] if args.mode=='full' else '-'} "
              f"failure={info['failure_type'] or '-'} takeovers={st['takeovers']} att_frac={st['attention_frac']:.2f}")
    dt = time.time() - t0
    print(f"\nsuccess {succ}/{args.episodes} | attention frac {att / steps:.2f} | takeovers {takeovers} "
          f"| {steps / dt:.0f} env steps/s")
    if frames:
        from PIL import Image

        Image.fromarray(np.concatenate(frames[:6], axis=0)).save(args.save_frames)
        print("saved", args.save_frames)


if __name__ == "__main__":
    main()
