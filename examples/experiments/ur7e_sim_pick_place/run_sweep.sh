#!/bin/bash
# Launch one (learner, actor) pair per (condition, seed) on the GPU server.
# Each pair gets its own port pair, checkpoint dir and GPU. Edit CONDITIONS / SEEDS / GPUS.
#   usage: bash run_sweep.sh <demo_path.pkl>
set -e
DEMO=${1:?demo path}
CONDITIONS=("full" "off" "alert:random" "alert:q_std")
SEEDS=(0 1 2)
GPUS=(0 1 2 3 4 5 6 7)
BUDGET=${BUDGET:-0.2}
MAX_STEPS_FLAG=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p ../../sweeps/logs
i=0
for cond in "${CONDITIONS[@]}"; do
  mode=${cond%%:*}; risk=${cond#*:}; [ "$risk" == "$cond" ] && risk=random
  for seed in "${SEEDS[@]}"; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}; port=$((6000 + 2 * i))
    name=${mode}_${risk}_b${BUDGET}_s${seed}
    ckpt=../../sweeps/${name}
    CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=.25 nohup python ../../train_rlpd.py \
      --exp_name=ur7e_sim_pick_place --learner --demo_path="$DEMO" --checkpoint_path=$ckpt \
      --seed=$seed --port=$port > ../../sweeps/logs/${name}_learner.log 2>&1 &
    sleep 5
    CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=.08 nohup python ../../train_rlpd.py \
      --exp_name=ur7e_sim_pick_place --actor --checkpoint_path=$ckpt --seed=$seed --port=$port \
      --human_mode=$mode --alert_risk=$risk --attention_budget=$BUDGET \
      > ../../sweeps/logs/${name}_actor.log 2>&1 &
    i=$((i + 1))
  done
done
echo "launched $i runs; logs in ../../sweeps/logs"
