#!/usr/bin/env bash
set -uo pipefail

if [ $# -eq 0 ]; then
    set -- 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2
fi
STEPS=${STEPS:-1000}
WARMUP=${WARMUP:-100}
BATCH=${BATCH:-32}
EVAL_INTERVAL=${EVAL_INTERVAL:-250}

for lr in "$@"; do
    echo "=== lr $lr ($STEPS steps, batch $BATCH)"
    python -m cs336_basics.train \
        --data_dir data/tinystories --device cuda --autocast_dtype bfloat16 \
        --batch_size "$BATCH" --context_length 256 --max_iters "$STEPS" --warmup_iters "$WARMUP" \
        --max_lr "$lr" --min_lr 0 --adam_beta2 0.95 \
        --eval_interval "$EVAL_INTERVAL" --eval_iters 50 --save_interval 100000 \
        --out_dir "runs/sweep_lr/lr-$lr-b$BATCH-s$STEPS" \
        --wandb_run_name "lr-$lr-b$BATCH-s$STEPS" \
        || echo "lr $lr failed"
done
