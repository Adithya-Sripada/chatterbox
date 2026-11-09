#!/usr/bin/env bash
# Step 3: LoRA fine-tune the base model on your chat data.
#   ./train.sh            uses the iteration count build_dataset.py recommended
#   ./train.sh 800        override iterations
#   ./train.sh eval       compare test-set loss: base model vs. your adapter
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
MODEL=$($PY -c 'import tomllib;print(tomllib.load(open("config.toml","rb"))["model"])')
ADAPTERS=$($PY -c 'import tomllib;print(tomllib.load(open("config.toml","rb")).get("adapter_path","adapters"))')
DATA=data/processed

if [[ ! -f $DATA/train.jsonl ]]; then
  echo "Run python src/parse_logs.py and python src/build_dataset.py first." >&2; exit 1
fi

if [[ "${1:-}" == "eval" ]]; then
  echo "== Base model (no adapter) =="
  $PY -m mlx_lm lora --model "$MODEL" --data $DATA --test --mask-prompt --test-batches -1 --batch-size 4 --adapter-path "" 2>&1 | grep -E "Test loss|Test ppl" || true
  echo "== With your adapter =="
  $PY -m mlx_lm lora --model "$MODEL" --data $DATA --test --mask-prompt --test-batches -1 --batch-size 4 --adapter-path "$ADAPTERS" 2>&1 | grep -E "Test loss|Test ppl" || true
  exit 0
fi

ITERS=${1:-$($PY -c 'import json;print(json.load(open("data/processed/meta.json"))["recommended_iters"])')}
EVAL_EVERY=$(( ITERS / 8 > 25 ? ITERS / 8 : 25 ))

echo "Training $MODEL for $ITERS iterations (validation every $EVAL_EVERY)..."
$PY -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data $DATA \
  --adapter-path "$ADAPTERS" \
  --mask-prompt \
  --batch-size 4 \
  --num-layers 16 \
  --learning-rate 3e-5 \
  --iters "$ITERS" \
  --steps-per-eval "$EVAL_EVERY" \
  --save-every "$EVAL_EVERY" \
  --val-batches -1 \
  --grad-checkpoint \
  --test

cat <<EOF

Done. Adapter saved in $ADAPTERS/ (checkpoints: $ADAPTERS/*_adapters.safetensors).
If validation loss went up near the end, copy the checkpoint with the lowest
"Val loss" over $ADAPTERS/adapters.safetensors.
Try it:  .venv/bin/python src/try_it.py --channel <channel>
EOF
