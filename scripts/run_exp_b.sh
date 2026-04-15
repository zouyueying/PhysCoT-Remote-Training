#!/bin/bash
# Experiment B: Full Fine-tune SFT via LLaMA-Factory (8x H100)
# No recon loss, physics tokens as format placeholders
set -e

cd "$(dirname "$0")/.."
echo "=== Experiment B: Full FT SFT (LLaMA-Factory) ==="
echo "GPUs: $(nvidia-smi -L | wc -l)"

mkdir -p logs
LOG="logs/exp_b_$(date +%Y%m%d_%H%M%S).log"

# Verify data
python3 -c "
import json, os
data = json.load(open('train/data/ladm_physcot.json'))
print(f'Training samples: {len(data)}')
img = data[0]['images'][0]
assert os.path.exists(img), f'Image not found: {img}. Run remap_paths.sh first!'
print(f'Data check OK: {img}')
"

export FORCE_TORCHRUN=1
export PYTHONWARNINGS="ignore"

# wandb auto-enable: turn on iff caller has WANDB_API_KEY in env OR has run
# `wandb login` before (which writes ~/.netrc). Otherwise stay silent.
REPORT_TO="none"
if [ -n "$WANDB_API_KEY" ]; then
    REPORT_TO="wandb"
    echo "[wandb] enabled via WANDB_API_KEY env var"
elif [ -f "$HOME/.netrc" ] && grep -q "api.wandb.ai" "$HOME/.netrc"; then
    REPORT_TO="wandb"
    echo "[wandb] enabled via ~/.netrc credentials"
else
    echo "[wandb] disabled (no WANDB_API_KEY, no ~/.netrc). Run 'wandb login' to enable."
fi
if [ "$REPORT_TO" = "wandb" ]; then
    export WANDB_PROJECT="${WANDB_PROJECT:-physcot}"
    export WANDB_NAME="${WANDB_NAME:-expb_$(date +%m%d_%H%M)}"
    export WANDB_DIR="${WANDB_DIR:-./logs/wandb}"
    mkdir -p "$WANDB_DIR"
fi

echo "Starting at $(date), log: $LOG"
llamafactory-cli train configs/stage2_full_ft.yaml --report_to "$REPORT_TO" 2>&1 | tee "$LOG"
echo "=== Experiment B complete at $(date) ==="
