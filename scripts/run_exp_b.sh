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

echo "Starting at $(date), log: $LOG"
llamafactory-cli train configs/stage2_full_ft.yaml 2>&1 | tee "$LOG"
echo "=== Experiment B complete at $(date) ==="
