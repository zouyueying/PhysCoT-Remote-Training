"""Stage 2 Joint Trainer: Full Fine-tune + LM Loss + Reconstruction Loss.

Multi-GPU version using Accelerate + DeepSpeed ZeRO-3.
No LoRA — all language model parameters are trainable.
Decoder is a small side module (~5MB), replicated on each GPU.

L_total = L_LM + λ_recon × L_recon

Usage (8x H100):
    accelerate launch --config_file configs/accelerate_ds_z3.yaml \
        -m physcot.train.stage2_joint_fullft \
        --config configs/stage2_full_ft_joint.yaml

Single GPU (debug):
    CUDA_VISIBLE_DEVICES=0 python -m physcot.train.stage2_joint_fullft \
        --config configs/stage2_full_ft_joint.yaml --max_steps 5
"""

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Stage2JointConfig:
    # model
    model_name_or_path: str = "checkpoints/stage1_merged_fixed"
    torch_dtype: str = "bfloat16"

    # Stage 1 checkpoint (for decoder.pt)
    stage1_ckpt: str = "checkpoints/stage1/checkpoint-epoch3"

    # projection hidden dim (must match Stage 1)
    proj_hidden_dim: int = 1024

    # data
    json_path: str = "train/data/ladm_physcot.json"
    features_dir: str = "features"
    max_length: int = 2048
    require_features: bool = True

    # training
    batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    bf16: bool = True
    max_steps: int = -1  # -1 = full training

    # loss weights
    lambda_recon: float = 0.1
    lambda_flow: float = 1.0
    lambda_depth: float = 1.0
    lambda_track: float = 1.0

    # output
    save_dir: str = "checkpoints/stage2_full_ft_joint"
    save_steps: int = 500
    logging_steps: int = 10

    # freeze
    freeze_vision_tower: bool = True
    freeze_mm_projector: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "Stage2JointConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        flat = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict):
                    flat.update(v)
                else:
                    flat[k] = v
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Training step (same logic as LoRA version, but model is unwrapped)
# ---------------------------------------------------------------------------
def training_step(model, decoder, batch, config, accelerator, phys_config):
    """Single forward pass → LM loss + reconstruction loss."""
    from physcot.tokens.projection import pool_flow, pool_depth, pool_track

    device = accelerator.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    flow_mask = batch["flow_token_mask"].to(device)
    depth_mask = batch["depth_token_mask"].to(device)
    track_mask = batch["track_token_mask"].to(device)

    n_flow = flow_mask.sum().item()
    n_depth = depth_mask.sum().item()
    n_track = track_mask.sum().item()
    has_physics = (n_flow + n_depth + n_track) > 0

    compute_dtype = torch.bfloat16 if config.bf16 else torch.float32
    flow_feat = batch["flow_features"].to(device, dtype=compute_dtype) if batch["flow_features"] is not None else None
    depth_feat = batch["depth_features"].to(device, dtype=compute_dtype) if batch["depth_features"] is not None else None
    track_feat = batch["track_features"].to(device, dtype=compute_dtype) if batch["track_features"] is not None else None

    # ========== 1. Forward pass ==========
    # With DeepSpeed ZeRO-3, use accelerator.unwrap_model for direct access
    unwrapped = accelerator.unwrap_model(model)

    # Access the language model backbone
    if hasattr(unwrapped, "model"):
        inner_model = unwrapped.model
    else:
        inner_model = unwrapped

    embed_layer = inner_model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids)

    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, 1, -1)

    lm_backbone = inner_model.language_model if hasattr(inner_model, "language_model") else inner_model
    outputs = lm_backbone(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state

    # ========== 2. LM loss ==========
    lm_head = unwrapped.lm_head if hasattr(unwrapped, "lm_head") else inner_model.lm_head
    logits = lm_head(hidden_states)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    lm_loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # ========== 3. Reconstruction loss ==========
    recon_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    result = {"flow_recon": 0.0, "depth_recon": 0.0, "track_recon": 0.0}

    if has_physics:
        mse = nn.functional.mse_loss
        for tok_type, mask, feat, pool_fn, lam in [
            ("flow", flow_mask, flow_feat, pool_flow, config.lambda_flow),
            ("depth", depth_mask, depth_feat, pool_depth, config.lambda_depth),
            ("track", track_mask, track_feat, pool_track, config.lambda_track),
        ]:
            n_tok = mask.sum().item()
            if n_tok == 0 or feat is None:
                continue
            h_all = hidden_states[mask]
            n_cfg = getattr(phys_config, f"{tok_type}_tokens")
            if n_tok > n_cfg:
                n_groups = n_tok // n_cfg
                usable = n_groups * n_cfg
                h_grouped = h_all[:usable].view(n_groups, n_cfg, -1)
                h = h_grouped.mean(dim=0, keepdim=True)
            else:
                h = h_all.unsqueeze(0)

            gt_pooled = pool_fn(feat)
            dec = getattr(decoder, f"{tok_type}_dec")
            # Decoder is fp32 for recon-loss numerical stability; hidden_states
            # are bf16 when model is loaded in bf16. Cast to match.
            dec_dtype = next(dec.parameters()).dtype
            pred = dec(h.to(dec_dtype))
            tok_recon = mse(pred.float(), gt_pooled.float())
            recon_loss = recon_loss + lam * tok_recon
            result[f"{tok_type}_recon"] = tok_recon.item()

    # ========== 4. Combined loss ==========
    total_loss = lm_loss + config.lambda_recon * recon_loss
    result["loss"] = total_loss
    result["lm_loss"] = lm_loss.item()
    result["recon_loss"] = recon_loss.item()
    return result


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_fn(batch):
    assert len(batch) == 1
    item = batch[0]
    return {
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "labels": item["labels"].unsqueeze(0),
        "flow_features": item["flow_features"].unsqueeze(0) if item["flow_features"] is not None else None,
        "depth_features": item["depth_features"].unsqueeze(0) if item["depth_features"] is not None else None,
        "track_features": item["track_features"].unsqueeze(0) if item["track_features"] is not None else None,
        "physics_token_mask": item["physics_token_mask"].unsqueeze(0),
        "flow_token_mask": item["flow_token_mask"].unsqueeze(0),
        "depth_token_mask": item["depth_token_mask"].unsqueeze(0),
        "track_token_mask": item["track_token_mask"].unsqueeze(0),
        "video_id": item["video_id"],
    }


# ---------------------------------------------------------------------------
# Build model (full fine-tune, no LoRA)
# ---------------------------------------------------------------------------
def build_model(config: Stage2JointConfig):
    """Load model for full fine-tune. No PEFT/LoRA."""
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    from physcot.tokens.physics_token import PhysicsTokenConfig
    from physcot.models.decoders import PhysicsDecoderBundle

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    model_path = config.model_name_or_path
    logger.info(f"Loading model: {model_path}")

    # Load model WITHOUT device_map — Accelerate/DeepSpeed handles placement
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Freeze vision tower and mm projector (same as Skyra)
    if config.freeze_vision_tower:
        if hasattr(model.model, "visual"):
            for p in model.model.visual.parameters():
                p.requires_grad = False
            logger.info("Froze vision_tower")

    if config.freeze_mm_projector:
        if hasattr(model.model, "multi_modal_projector"):
            for p in model.model.multi_modal_projector.parameters():
                p.requires_grad = False
            logger.info("Froze multi_modal_projector")

    # Verify physics token embeddings are non-zero
    phys_ids = [151665, 151666, 151667]
    embed_weight = model.model.get_input_embeddings().weight
    for pid in phys_ids:
        if pid < embed_weight.shape[0]:
            norm = embed_weight[pid].float().norm().item()
            logger.info(f"Physics token {pid} embedding norm: {norm:.4f}")
            assert norm > 0.01, f"Physics token {pid} has zero embedding! Use stage1_merged_fixed."

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # Load decoder (small, replicated on each GPU)
    phys_config = PhysicsTokenConfig()
    decoder = PhysicsDecoderBundle(phys_config, hidden_dim=config.proj_hidden_dim)

    stage1_path = config.stage1_ckpt
    dec_path = os.path.join(stage1_path, "decoder.pt")
    if os.path.exists(dec_path):
        decoder.load_state_dict(torch.load(dec_path, map_location="cpu", weights_only=True))
        logger.info(f"Loaded decoder from {dec_path}")
    else:
        logger.warning(f"decoder.pt not found at {dec_path}, using random init")

    return model, tokenizer, decoder, phys_config


# ---------------------------------------------------------------------------
# Main training loop with Accelerate
# ---------------------------------------------------------------------------
def train(config: Stage2JointConfig):
    from accelerate import Accelerator
    from physcot.data.dataset import PhysCoTDataset
    from physcot.tokens.physics_token import PhysicsTokenConfig

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.bf16 else "no",
    )

    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    model, tokenizer, decoder, phys_config = build_model(config)

    dataset = PhysCoTDataset(
        json_path=config.json_path,
        tokenizer=tokenizer,
        features_dir=config.features_dir,
        config=phys_config,
        max_length=config.max_length,
        require_features=config.require_features,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    total_steps_per_epoch = len(dataloader) // config.gradient_accumulation_steps
    num_update_steps = total_steps_per_epoch * config.num_epochs
    if config.max_steps > 0:
        num_update_steps = min(num_update_steps, config.max_steps)

    # Optimizer: model params + decoder params
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": config.learning_rate},
            {"params": decoder.parameters(), "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )

    # Cosine scheduler with linear warmup
    num_warmup = int(num_update_steps * config.warmup_ratio)
    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        progress = float(step - num_warmup) / float(max(1, num_update_steps - num_warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare with Accelerate (handles DeepSpeed ZeRO-3 wrapping)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Decoder is small — move to device, NOT wrapped by DeepSpeed
    decoder = decoder.to(accelerator.device)
    decoder.train()

    os.makedirs(config.save_dir, exist_ok=True)
    global_step = 0
    accum_loss, accum_lm, accum_recon, accum_count = 0.0, 0.0, 0.0, 0

    logger.info(f"Training: epochs={config.num_epochs}, lr={config.learning_rate}, "
                f"lambda_recon={config.lambda_recon}, total_steps={num_update_steps}")
    logger.info(f"GPUs: {accelerator.num_processes}, grad_accum: {config.gradient_accumulation_steps}")

    start_time = time.time()

    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        model.train()

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                result = training_step(model, decoder, batch, config, accelerator, phys_config)
                loss = result["loss"]

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Clip gradients for model (decoder grads clipped separately)
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    nn.utils.clip_grad_norm_(decoder.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            accum_loss += result["lm_loss"] + config.lambda_recon * result["recon_loss"]
            accum_lm += result["lm_loss"]
            accum_recon += result["recon_loss"]
            accum_count += 1

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % config.logging_steps == 0 and accelerator.is_main_process:
                    avg_loss = accum_loss / max(accum_count, 1)
                    avg_lm = accum_lm / max(accum_count, 1)
                    avg_recon = accum_recon / max(accum_count, 1)
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    eta = elapsed / max(global_step, 1) * (num_update_steps - global_step)
                    logger.info(
                        f"  step={global_step}/{num_update_steps}  "
                        f"loss={avg_loss:.4f}  lm={avg_lm:.4f}  recon={avg_recon:.4f}  "
                        f"(flow={result['flow_recon']:.4f} depth={result['depth_recon']:.4f} "
                        f"track={result['track_recon']:.4f})  "
                        f"lr={lr:.2e}  eta={eta/3600:.1f}h"
                    )
                    accum_loss, accum_lm, accum_recon, accum_count = 0.0, 0.0, 0.0, 0

                if global_step % config.save_steps == 0:
                    _save(accelerator, model, decoder, tokenizer, config, f"step{global_step}")

                if config.max_steps > 0 and global_step >= config.max_steps:
                    break

        # Save at end of each epoch
        _save(accelerator, model, decoder, tokenizer, config, f"epoch{epoch+1}")

        if config.max_steps > 0 and global_step >= config.max_steps:
            break

    # Final save
    _save(accelerator, model, decoder, tokenizer, config, "final")
    logger.info("Training complete.")


def _save(accelerator, model, decoder, tokenizer, config, tag):
    """Save checkpoint using Accelerate (handles ZeRO-3 gathering)."""
    save_path = os.path.join(config.save_dir, f"checkpoint-{tag}")

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)

    # save_model handles ZeRO-3 weight gathering automatically
    accelerator.save_model(unwrapped, save_path, max_shard_size="5GB")

    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_path)
        torch.save(decoder.state_dict(), os.path.join(save_path, "decoder.pt"))
        logger.info(f"Checkpoint saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=-1, help="Override max_steps for debug")
    args = parser.parse_args()

    config = Stage2JointConfig.from_yaml(args.config)
    if args.max_steps > 0:
        config.max_steps = args.max_steps

    train(config)


if __name__ == "__main__":
    main()
