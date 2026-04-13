"""PhysCoTDataset: loads video frames + physics features + CoT for Stage 1/2 training.

Each sample yields:
    - input_ids, attention_mask, labels  (tokenized conversation with physics tokens)
    - flow_features, depth_features, track_features  (.pt tensors, may be None)
    - dino_features, freq_features      (v8 5-type, may be None — lenient mode)
    - physics_token_mask  (bool mask over input_ids marking physics token positions)
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from physcot.tokens.physics_token import (
    ALL_PHYSICS_TOKENS,
    DEPTH_TOKEN,
    DINO_TOKEN,
    FLOW_TOKEN,
    FREQ_TOKEN,
    TRACK_TOKEN,
    PhysicsTokenConfig,
)


# v8 — frequency energy scaling constant.
# Raw freq_energy values fall in [0.0003, 0.0288] (corpus stats from 50-sample
# inspection on 2026-04-10). With random Linear init, the projection's first
# layer would produce near-zero outputs and starve gradient flow. Multiplying
# by 200 brings values into [~0.06, ~5.76], comparable to z-scored dino input.
# This is documented in agent_docs/gotchas.md problem 13 (v8 freq normalization).
FREQ_SCALE = 200.0


def derive_video_id(image_path: str) -> str:
    """Derive video_id from an image path.

    E.g. '.../parsed_frames/Panda-70M/fake/Gen/vid-0/1.png'
         -> 'Panda-70M_fake_Gen_vid-0'
    """
    marker = "/parsed_frames/"
    idx = image_path.find(marker)
    if idx == -1:
        raise ValueError(f"Cannot find '{marker}' in image path: {image_path}")
    rel = image_path[idx + len(marker):]       # 'Panda-70M/fake/Gen/vid-0/1.png'
    video_dir = os.path.dirname(rel)            # 'Panda-70M/fake/Gen/vid-0'
    return video_dir.replace("/", "_")          # 'Panda-70M_fake_Gen_vid-0'


def normalize_flow(flow: torch.Tensor) -> torch.Tensor:
    """Normalize optical flow to [-1, 1] via per-video max-abs scaling.

    Raw RAFT flow values can reach ±1700, causing MSE loss to dominate
    over depth (~[0,7]) and track (~[0,1]) features. This brings flow
    into a comparable range while preserving relative motion patterns.
    """
    max_abs = flow.abs().max()
    if max_abs > 0:
        flow = flow / max_abs
    return flow


def load_feature(features_dir: str, feat_type: str, video_id: str) -> Optional[torch.Tensor]:
    """Load a physics feature tensor, returning None if file missing."""
    path = os.path.join(features_dir, feat_type, f"{video_id}.pt")
    if os.path.exists(path):
        t = torch.load(path, map_location="cpu", weights_only=True)
        if feat_type == "flow":
            t = normalize_flow(t)
        return t
    return None


def normalize_dino(dino_diff_raw: torch.Tensor) -> torch.Tensor:
    """Normalize raw dino_diff (T-1, 768) to a per-sample z-scored vector (768,).

    Steps:
        1. Mean-collapse temporal dim: (T-1, 768) → (768,)
        2. Z-score across the 768-dim feature axis (μ=0, σ=1)

    The z-score is computed per-sample because:
    - DINO embeddings already have implicit normalization from the backbone
    - Per-sample z-score handles the small inter-sample variance
      (corpus stats showed dino mean ∈ [-0.0016, 0.0022])
    """
    if dino_diff_raw.dim() != 2 or dino_diff_raw.shape[-1] != 768:
        raise ValueError(
            f"normalize_dino expects (T-1, 768), got {tuple(dino_diff_raw.shape)}"
        )
    v = dino_diff_raw.float().mean(dim=0)                        # (768,)
    v = (v - v.mean()) / (v.std() + 1e-8)                        # z-score
    return v


def normalize_freq(freq_energy_raw: torch.Tensor) -> torch.Tensor:
    """Normalize raw freq_energy (T,) to a single scalar feature (1,).

    Steps:
        1. Mean-collapse temporal dim: (T,) → scalar
        2. Multiply by FREQ_SCALE (=200) to bring small values into projection-friendly range

    Why scale and not normalize:
    - freq_energy ∈ [0.0003, 0.0288] (corpus stats from 50-sample inspection)
    - Per-sample z-score across T frames gives ~0 (intra-video variance is tiny)
    - Corpus-level normalization would need offline pass; FREQ_SCALE=200 is a
      pragmatic constant that brings values into [0.06, 5.76] range, comparable
      to z-scored dino. Refine if Stage 1 doesn't converge.
    """
    if freq_energy_raw.dim() != 1:
        raise ValueError(
            f"normalize_freq expects (T,), got {tuple(freq_energy_raw.shape)}"
        )
    v = freq_energy_raw.float().mean() * FREQ_SCALE              # scalar
    return v.unsqueeze(0)                                         # (1,)


def load_dino_freq(
    features_dir: str,
    video_id: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load and normalize dino_diff + freq_energy from a single .pt file.

    Unlike flow/depth/track which each have their own subdirectory, dino and
    freq are stored together in features/dino_freq/{video_id}.pt as a dict
    with keys 'dino_diff' (shape T-1, 768) and 'freq_energy' (shape T,).

    Returns:
        (dino_normalized, freq_normalized) — each (768,) and (1,) tensors,
        or (None, None) if file missing (lenient mode).

    Note: 8679/17913 ≈ 48% of videos have this file. Missing files do not
    block training of flow/depth/track loss for the same sample.
    """
    path = os.path.join(features_dir, "dino_freq", f"{video_id}.pt")
    if not os.path.exists(path):
        return None, None
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        dino_norm = normalize_dino(d["dino_diff"])
        freq_norm = normalize_freq(d["freq_energy"])
        return dino_norm, freq_norm
    except Exception:
        # Corrupted file or unexpected format — treat as missing
        return None, None


class PhysCoTDataset(Dataset):
    """Dataset for PhysCoT Stage 1 (alignment) and Stage 2 (SFT) training.

    Args:
        json_path: Path to ladm_physcot.json (or ladm_sft_local.json).
        tokenizer: HuggingFace tokenizer with physics tokens registered.
        features_dir: Root dir containing flow/, depth/, track/ subdirs
                      (and optionally dino_freq/ for v8 5-type training).
        config: PhysicsTokenConfig — controls which token types are loaded.
                v5 default: dino_tokens=0, freq_tokens=0 → skip dino_freq/ loading.
                v8: dino_tokens=4, freq_tokens=4 → load dino_freq/ when available
                (lenient: missing files do NOT exclude the sample, only the sample's
                dino/freq loss is skipped).
        max_length: Max token sequence length (truncation).
        require_features: If True, skip samples missing flow/depth/track files.
                          Note: dino/freq are NEVER required (lenient mode by design).
    """

    def __init__(
        self,
        json_path: str,
        tokenizer,
        features_dir: str,
        config: PhysicsTokenConfig = None,
        max_length: int = 2048,
        require_features: bool = True,
    ):
        self.tokenizer = tokenizer
        self.features_dir = features_dir
        self.config = config or PhysicsTokenConfig()
        self.max_length = max_length

        # v8: lazy loading toggles based on config
        self.load_dino = self.config.dino_tokens > 0
        self.load_freq = self.config.freq_tokens > 0

        with open(json_path, "r") as f:
            raw_data = json.load(f)

        # Build samples, optionally filtering by feature availability.
        # Required features = flow + depth + track (the original 3).
        # dino/freq are NEVER part of the require check (lenient mode).
        self.samples = []
        for entry in raw_data:
            video_id = derive_video_id(entry["images"][0])
            if require_features:
                has_all = all(
                    os.path.exists(os.path.join(features_dir, ft, f"{video_id}.pt"))
                    for ft in ("flow", "depth", "track")
                )
                if not has_all:
                    continue
            self.samples.append({
                "messages": entry["messages"],
                "images": entry["images"],
                "video_id": video_id,
            })

        # Cache physics token IDs (skip tokens not in vocab, e.g. dino/freq in 3-type base)
        self.physics_token_ids = {}
        for tok in ALL_PHYSICS_TOKENS:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                self.physics_token_ids[tok] = tid

    def __len__(self) -> int:
        return len(self.samples)

    def _build_text(self, messages: List[dict]) -> Tuple[str, str]:
        """Build prompt (system+user) and completion (assistant) text.

        Returns:
            (prompt_text, completion_text)
        """
        prompt_parts = []
        completion = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                completion = content
        prompt = "".join(prompt_parts) + "<|im_start|>assistant\n"
        return prompt, completion

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        video_id = sample["video_id"]

        # --- Tokenize conversation ---
        prompt_text, completion_text = self._build_text(sample["messages"])

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(
            completion_text + "<|im_end|>", add_special_tokens=False
        )

        input_ids = prompt_ids + completion_ids

        # Labels: mask prompt with -100, keep completion
        labels = [-100] * len(prompt_ids) + completion_ids.copy()

        # Truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # --- Physics token position masks ---
        # Bool mask per token type: True at positions where that token appears
        physics_token_masks = {}
        for tok_str, tok_id in self.physics_token_ids.items():
            physics_token_masks[tok_str] = (input_ids == tok_id)

        # Combined mask (any physics token)
        physics_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for mask in physics_token_masks.values():
            physics_token_mask |= mask

        # --- Load physics features ---
        flow_feat = load_feature(self.features_dir, "flow", video_id)
        depth_feat = load_feature(self.features_dir, "depth", video_id)
        track_feat = load_feature(self.features_dir, "track", video_id)

        # v8: optionally load dino/freq (single file with both, lenient if missing)
        dino_feat: Optional[torch.Tensor] = None
        freq_feat: Optional[torch.Tensor] = None
        if self.load_dino or self.load_freq:
            dino_loaded, freq_loaded = load_dino_freq(self.features_dir, video_id)
            if self.load_dino:
                dino_feat = dino_loaded
            if self.load_freq:
                freq_feat = freq_loaded

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "flow_features": flow_feat,
            "depth_features": depth_feat,
            "track_features": track_feat,
            "dino_features": dino_feat,                          # v8 (None if v5 or missing)
            "freq_features": freq_feat,                          # v8 (None if v5 or missing)
            "physics_token_mask": physics_token_mask,
            "flow_token_mask": physics_token_masks.get(FLOW_TOKEN, torch.zeros_like(input_ids, dtype=torch.bool)),
            "depth_token_mask": physics_token_masks.get(DEPTH_TOKEN, torch.zeros_like(input_ids, dtype=torch.bool)),
            "track_token_mask": physics_token_masks.get(TRACK_TOKEN, torch.zeros_like(input_ids, dtype=torch.bool)),
            "dino_token_mask": physics_token_masks.get(DINO_TOKEN, torch.zeros_like(input_ids, dtype=torch.bool)),
            "freq_token_mask": physics_token_masks.get(FREQ_TOKEN, torch.zeros_like(input_ids, dtype=torch.bool)),
            "video_id": video_id,
        }
