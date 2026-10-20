#!/usr/bin/env bash
set -uo pipefail

BATCH=$1
LR=$2
TOTAL_TOKENS=${TOTAL_TOKENS:-327680000}
STEPS=${STEPS:-$((TOTAL_TOKENS / (BATCH * 256)))}
WARMUP=$((STEPS / 20))
EVAL_INTERVAL=$((STEPS / 20))
LOG_INTERVAL=$((STEPS / 500 > 10 ? STEPS / 500 : 10))

echo "=== batch $BATCH, lr $LR, $STEPS steps"
python -m cs336_basics.train \
    --data_dir data/tinystories --device cuda --autocast_dtype bfloat16 \
    --batch_size "$BATCH" --context_length 256 --max_iters "$STEPS" --warmup_iters "$WARMUP" \
    --max_lr "$LR" --min_lr 0 --adam_beta2 0.95 \
    --eval_interval "$EVAL_INTERVAL" --eval_iters 50 --eval_batch_size 128 \
    --log_interval "$LOG_INTERVAL" --save_interval 100000000 \
    --out_dir "runs/sweep_batch/b$BATCH-lr$LR" \
    --wandb_run_name "batch-$BATCH-lr$LR-s$STEPS"
