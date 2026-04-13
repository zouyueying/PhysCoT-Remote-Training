#!/bin/bash
# Experiment C: Full Fine-tune + Joint LM + Recon Loss (8x H100)
# Physics tokens with real semantic grounding via reconstruction loss
set -e

cd "$(dirname "$0")/.."
echo "=== Experiment C: Full FT + Joint Loss (Accelerate + DeepSpeed) ==="
echo "GPUs: $(nvidia-smi -L | wc -l)"

mkdir -p logs
LOG="logs/exp_c_$(date +%Y%m%d_%H%M%S).log"

# Verify data + features
python3 -c "
import json, os
data = json.load(open('train/data/ladm_physcot.json'))
print(f'Training samples: {len(data)}')
img = data[0]['images'][0]
assert os.path.exists(img), f'Image not found: {img}. Run remap_paths.sh first!'

# Check features
for feat_type in ['flow', 'depth', 'track']:
    feat_dir = f'features/{feat_type}'
    if os.path.isdir(feat_dir):
        n = len([f for f in os.listdir(feat_dir) if f.endswith('.pt')])
        print(f'{feat_type} features: {n} files')
    else:
        print(f'WARNING: {feat_dir} not found! Recon loss will be zero.')
print('Data check OK')
"

echo "Starting at $(date), log: $LOG"
accelerate launch \
    --config_file configs/accelerate_ds_z3.yaml \
    -m physcot.train.stage2_joint_fullft \
    --config configs/stage2_full_ft_joint.yaml \
    2>&1 | tee "$LOG"
echo "=== Experiment C complete at $(date) ==="
