"""Physics token configuration and special token definitions for PhysCoT.

Defines the five physics token types (flow, depth, track, dino, freq) and
provides utilities for registering them as special tokens in the tokenizer.

Token type expansion (v8 / IB selection):
  - flow / depth / track: original 3 types (v5)
  - dino / freq: added in v8 to cover semantic and frequency anomalies
                 (Identity Inconsistency, Color Over-saturation, GAN artifacts,
                  Unnatural Blur, Texture Jittering)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Special token strings used in CoT text and tokenizer
FLOW_TOKEN = "<flow_tok>"
DEPTH_TOKEN = "<depth_tok>"
TRACK_TOKEN = "<track_tok>"
DINO_TOKEN = "<dino_tok>"
FREQ_TOKEN = "<freq_tok>"

ALL_PHYSICS_TOKENS = [FLOW_TOKEN, DEPTH_TOKEN, TRACK_TOKEN, DINO_TOKEN, FREQ_TOKEN]


@dataclass
class PhysicsTokenConfig:
    """Configuration for physics tokens in PhysCoT.

    Attributes:
        flow_tokens: Number of flow tokens per insertion (default 4).
        depth_tokens: Number of depth tokens per insertion (default 4).
        track_tokens: Number of track tokens per insertion (default 4).
        dino_tokens: Number of DINO semantic tokens per insertion (default 4).
        freq_tokens: Number of FFT frequency tokens per insertion (default 4).
        token_dim: Must match Qwen2.5-VL-7B hidden size (3584).
        flow_feature_dim: RAFT output channels (u, v) = 2.
        depth_feature_dim: DepthAnything frame-diff channels = 1.
        track_feature_dim: SAM2 trajectory feature dim (cx, cy, bw, bh) = 4.
        dino_feature_dim: DINOv2 CLS embedding dim = 768.
        freq_feature_dim: FFT high-frequency energy = 1 (scalar per frame).

    Notes:
        - DEFAULT IS v5 (3-type, total 12 with track=4). dino_tokens and
          freq_tokens default to 0, so any code that constructs
          PhysicsTokenConfig() without explicit args gets v5 backward-compat.
        - For v8 (5-type + IB selection), explicitly pass dino_tokens=4
          and freq_tokens=4 at instantiation site. Recommended via YAML config.
        - Track default is still 4 here; configs that need 8 (CLAUDE.md spec)
          must override at instantiation site.
    """

    flow_tokens: int = 4
    depth_tokens: int = 4
    track_tokens: int = 4
    dino_tokens: int = 0   # v8 opt-in: set to 4 in v8 configs
    freq_tokens: int = 0   # v8 opt-in: set to 4 in v8 configs
    token_dim: int = 3584  # Qwen2.5-VL-7B hidden size
    flow_feature_dim: int = 2
    depth_feature_dim: int = 1
    track_feature_dim: int = 4
    dino_feature_dim: int = 768
    freq_feature_dim: int = 1

    @property
    def total_tokens(self) -> int:
        return (
            self.flow_tokens
            + self.depth_tokens
            + self.track_tokens
            + self.dino_tokens
            + self.freq_tokens
        )

    @property
    def token_counts(self) -> Dict[str, int]:
        return {
            FLOW_TOKEN: self.flow_tokens,
            DEPTH_TOKEN: self.depth_tokens,
            TRACK_TOKEN: self.track_tokens,
            DINO_TOKEN: self.dino_tokens,
            FREQ_TOKEN: self.freq_tokens,
        }

    @property
    def feature_dims(self) -> Dict[str, int]:
        return {
            FLOW_TOKEN: self.flow_feature_dim,
            DEPTH_TOKEN: self.depth_feature_dim,
            TRACK_TOKEN: self.track_feature_dim,
            DINO_TOKEN: self.dino_feature_dim,
            FREQ_TOKEN: self.freq_feature_dim,
        }


def register_physics_tokens(tokenizer) -> Dict[str, int]:
    """Register physics special tokens in the tokenizer.

    Args:
        tokenizer: A HuggingFace tokenizer instance.

    Returns:
        Dict mapping token string to token ID.
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ALL_PHYSICS_TOKENS}
    )
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in ALL_PHYSICS_TOKENS}
    return token_ids


def get_physics_token_ids(tokenizer) -> Dict[str, int]:
    """Get token IDs for already-registered physics tokens.

    Raises ValueError if tokens are not registered.
    """
    token_ids = {}
    for tok in ALL_PHYSICS_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid == tokenizer.unk_token_id:
            raise ValueError(
                f"Physics token {tok} not registered in tokenizer. "
                f"Call register_physics_tokens() first."
            )
        token_ids[tok] = tid
    return token_ids


def build_physics_token_string(token_type: str, count: int) -> str:
    """Build a repeated token string for insertion into CoT text.

    Example: build_physics_token_string("<flow_tok>", 4) -> "<flow_tok><flow_tok><flow_tok><flow_tok>"
    """
    if token_type not in ALL_PHYSICS_TOKENS:
        raise ValueError(f"Unknown token type: {token_type}. Must be one of {ALL_PHYSICS_TOKENS}")
    return token_type * count
