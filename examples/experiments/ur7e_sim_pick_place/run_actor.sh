#!/bin/bash
# usage: bash run_actor.sh [--human_mode=full|alert|off] [--alert_risk=q_std|random|neg_q]
#                          [--attention_budget=0.2] [--ros2_bridge] [--norealtime] [--port=5588] [--seed=0]
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM:-.1}
python ../../train_rlpd.py \
    --exp_name=ur7e_sim_pick_place \
    --checkpoint_path=../../experiments/ur7e_sim_pick_place/debug \
    --actor \
    "$@"
