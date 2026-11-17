# serl_sim — UR7e simulation and simulated human for HIL-SERL

MuJoCo simulation used for the attention-budget research. Its interface matches `franka_env.FrankaEnv` exactly, so `train_rlpd.py` and all the HIL-SERL wrappers work unchanged. ROS2 is optional: use it for visualization and for taking over in sim with a real SpaceMouse.

| Module | Purpose |
|---|---|
| `envs/ur7e_pick_place_env.py` | UR7e (UR5e kinematics) + parallel gripper + pick-and-place. Cartesian impedance control (with Franka-style error clipping), external F/T, two cameras (front / wrist), automatic failure labels |
| `human/sim_human.py` | Simulated human: `off` / `full` (standard HIL-SERL, 100% attention) / `alert` (watches only when alerted; reaction delay and attention accounting) / `expert` (collects demos) |
| `attention.py` | Budgeted alert scheduler: rank-transformed risk + budget controller (long-run attention → b) + miss-rate controller |
| `ros2/bridge.py` | ROS2 Humble bridge: publishes state, images and attention status; spacenav teleop |
| `scripts/check_sim.py` | Sanity check / speed test / interactive viewer |
| `scripts/analyze_risk_log.py` | Offline analysis of the risk logs (oracle bound, AUROC, budget-matched recall, training curves) |

## Install (Ubuntu 22.04)

```bash
# 1) Set up the base env following the HIL-SERL README (conda, python 3.10, jax-cuda, serl_launcher, serl_robot_infra)
conda activate hilserl
# 2) Simulator
cd serl_sim && pip install -e .
sudo apt install -y libegl1 libgl1          # MuJoCo offscreen rendering (EGL is used automatically on headless servers)
python scripts/check_sim.py --mode expert --episodes 10          # expect ~100% success
python scripts/check_sim.py --mode expert --episodes 3 --viewer  # MuJoCo viewer (needs a display)
```

ROS2 (optional, only needed for `--ros2_bridge` / `--human_mode=ros2`):

```bash
sudo apt install -y ros-humble-desktop ros-humble-spacenav
# Order matters: source ROS2 first, then activate conda (conda must be python 3.10, matching Humble)
source /opt/ros/humble/setup.bash && conda activate hilserl
python -c "import rclpy; print('rclpy ok')"
# If you get a GLIBCXX / libstdc++ error:  conda install -c conda-forge libstdcxx-ng
```

## Usage (run from `examples/`)

```bash
cd examples
# 1. Expert demos (no human needed, a few minutes)
python record_sim_demos.py --exp_name=ur7e_sim_pick_place --successes_needed=20

# 2. Training (two terminals). Default is 10 Hz real time, same UTD ratio as the real robot
cd experiments/ur7e_sim_pick_place
bash run_learner.sh --demo_path=../../demo_data/<file>.pkl --debug
bash run_actor.sh --human_mode=full --debug                       # standard HIL-SERL (100% attention)
bash run_actor.sh --human_mode=alert --alert_risk=q_std --attention_budget=0.2 --debug
bash run_actor.sh --human_mode=off --debug                        # RLPD without a human

# 3. A real person intervening in sim (ROS2 + SpaceMouse)
ros2 run spacenav spacenav_node &
bash run_actor.sh --human_mode=ros2 --ros2_bridge
ros2 run rqt_image_view rqt_image_view /hil_sim/camera/front/image_raw
ros2 topic echo /hil_sim/human_state

# 4. Parallel sweep on the GPU server (one port pair + one checkpoint dir per run)
bash run_sweep.sh ../../demo_data/<file>.pkl

# 5. Analysis
python ../../../serl_sim/scripts/analyze_risk_log.py ../../sweeps/alert_q_std_b0.2_s0
```

Notes
- `--debug` disables wandb. Without it you need `wandb login`.
- For concurrent runs on one machine use different `--port` values (the actor and learner of one run must match); `run_sweep.sh` assigns them automatically.
- If the checkpoint directory already exists, `train_rlpd.py` waits for Enter (original HIL-SERL behaviour). Delete the old directory before starting a fresh run.
- `--norealtime` makes the actor run as fast as possible (~250 steps/s), but the UTD ratio then depends on GPU speed and is no longer comparable to the paper or the real robot. Use it only for debugging.
- Every `buffer_period` (1000) steps the actor writes `risk_log/risk_<step>.pkl`. Each step records: Q mean/std, risk, alert, τ, whether the human was attending, whether they intervened, success / failure / failure type.

## Simulated-human model (state it explicitly in the paper)

- `full`: the human watches every step. They take over after 3 consecutive steps of "policy direction opposes expert / policy stalls / wrong gripper action / object dropped", keep control at least 10 steps and until the current sub-skill (phase) finishes (at most 40).
- `alert`: away by default. After an alert: reaction 10–30 steps (counted as attention, no control) → watch 30 steps (same takeover rule as `full`) → away. An alert while watching extends the watch window.
- The expert is a privileged-state scripted policy with Gaussian noise of 0.05.

## Known limitations

- Arm links have no collision geometry (only the gripper, table and object collide). The safety box keeps the TCP above the table.
- The sim reward is ground-truth success, not the image classifier used on the real robot. Real-robot reward noise is not modelled.
- Not verified on Ubuntu (developed and tested on macOS + MuJoCo 4.0.1): the ROS2 bridge, the JAX training loop (`train_rlpd.py` changes) and EGL rendering.
