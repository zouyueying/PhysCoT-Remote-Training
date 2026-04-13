# PhysCoT Remote Training

Physics-grounded Chain-of-Thought for AIGC Video Detection.
Full fine-tune training on 8x H100.

## Quick Start

```bash
# 1. Install dependencies
bash scripts/setup_env.sh

# 2. Remap data paths to your server
bash scripts/remap_paths.sh /path/to/data_root

# 3. Run Experiment B (SFT, ~15h)
bash scripts/run_exp_b.sh

# 4. Run Experiment C (Joint Loss, ~15h)
bash scripts/run_exp_c.sh
```

## Data Layout

```
DATA_ROOT/
├── parsed_frames/                    # 17GB, video frames
│   ├── kinetics/
│   │   ├── fake/{generator}/{video_id}/1.png...16.png
│   │   └── real/{generator}/{video_id}/1.png...16.png
│   └── Panda-70M/
├── stage1_merged_fixed/              # 16GB, base model
│   ├── model-0000{1-4}-of-00004.safetensors
│   ├── config.json, tokenizer.json, ...
├── stage1/checkpoint-epoch3/         # decoder.pt (~5MB, for Exp C only)
│   └── decoder.pt
├── features/                         # ~10GB, for Exp C only
│   ├── flow/{video_id}.pt
│   ├── depth/{video_id}.pt
│   └── track/{video_id}.pt
└── physcot/                          # this repo
```

## Two Experiments

### Experiment B: Full FT SFT (LLaMA-Factory)

Physics tokens appear in CoT as format placeholders (no physics semantics).
Uses LLaMA-Factory framework, zero custom code.

- **Config**: `configs/stage2_full_ft.yaml`
- **Data**: `ladm_physcot.json` (4029 samples, Real:Fake = 1:1)
- **Base**: `stage1_merged_fixed` (Qwen2.5-VL-7B + 3 physics tokens)
- **Method**: Full fine-tune, freeze vision_tower + mm_projector
- **Hyperparams**: lr=1e-5, epochs=5, cosine schedule, DeepSpeed ZeRO-3

### Experiment C: Full FT + Joint Loss (Accelerate + DeepSpeed)

Physics tokens grounded in real physical features via reconstruction loss.
Custom trainer with dual loss: L = L_LM + 0.1 × L_recon.

- **Config**: `configs/stage2_full_ft_joint.yaml`
- **Data**: `ladm_physcot.json` + `features/*.pt`
- **Base**: `stage1_merged_fixed` + `decoder.pt` from Stage 1
- **Method**: Full fine-tune + MSE reconstruction on physics token hidden states
- **Requires**: `features/` directory with pre-extracted .pt files

### What the experiments answer

| Result | Meaning | Next step |
|--------|---------|-----------|
| B > 91% (Skyra) | Physics tokens help even without semantics | Paper story: format + CoT structure |
| B ≈ 91% | Physics tokens are neutral | Need Exp C to add value |
| C > B > 91% | Physics semantics add incremental value | Core PhysCoT contribution proven |
| C > 91% > B | Semantics essential, format alone insufficient | Strongest paper story |

## Evaluation

```bash
cd eval
python inference.py \
    --index_json /path/to/test_index_local.json \
    --model_path ../checkpoints/stage2_full_ft \
    --model_name PhysCoT-SFT-FullFT \
    --save_dir ../results/exp_b

python eval.py --json_file_path ../results/exp_b/PhysCoT-SFT-FullFT.json
```

## Hardware

- **Tested**: 8x H100 80GB, single node
- **Minimum**: 2x A100 80GB (adjust `num_processes` in accelerate config)
- **Training time**: ~15h per experiment on 8x H100
