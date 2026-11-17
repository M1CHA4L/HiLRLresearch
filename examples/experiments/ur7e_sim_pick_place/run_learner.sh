#!/bin/bash
# usage: bash run_learner.sh --demo_path=<pkl> [--checkpoint_path=...] [--port=5588] [--seed=0] [--debug]
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM:-.3}
python ../../train_rlpd.py \
    --exp_name=ur7e_sim_pick_place \
    --checkpoint_path=../../experiments/ur7e_sim_pick_place/debug \
    --learner \
    "$@"
