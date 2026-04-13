"""Physics feature → token embedding projection layers.

Each projection compresses a variable-size physics feature map to a fixed set
of token embeddings [n_tokens, token_dim] that replace the special token
embeddings in the LLM input sequence.

Data flow:
    GT feature (variable spatial) → AdaptivePool → compressed (fixed)
                                                  → MLP → [n_tokens, token_dim]

Token type coverage (v8):
    flow / depth / track : original 3 types (v5 baseline)
    dino  : DINOv2 CLS frame-diff (semantic / identity inconsistency)
    freq  : FFT high-frequency energy (GAN artifacts / blur / texture)

NOTE on dino/freq input convention:
    The dataset.py loader is expected to apply per-type normalization and
    temporal collapse BEFORE passing to projection:
        dino: raw (T-1, 768) → mean over T → (768,) → z-score normalize
        freq: raw (T,)       → mean over T → (1,)   → max-abs normalize to [-1, 1]
    So projection receives:
        dino: (B, 768) or (768,)
        freq: (B, 1)   or (1,) or scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from physcot.tokens.physics_token import PhysicsTokenConfig

# Fixed pool output sizes for each feature type.
# These define the "compressed feature" resolution used in both
# projection (input side) and decoder (reconstruction target).
FLOW_POOL_SIZE = (4, 8, 8)   # (T, H, W) → flat = 4*8*8*2 = 512
DEPTH_POOL_SIZE = (4, 8, 8)  # (T, H, W) → flat = 4*8*8*1 = 256
TRACK_POOL_T = 8              # T only   → flat = 8*4 = 32
DINO_FLAT = 768               # already mean-pooled in dataset → flat = 768
FREQ_FLAT = 1                 # already mean-pooled in dataset → flat = 1


def pool_flow(x: torch.Tensor) -> torch.Tensor:
    """Adaptive-pool flow features to fixed size.

    Args:
        x: (B, T, H, W, 2) or (T, H, W, 2) raw flow features.
    Returns:
        (B, 2, *FLOW_POOL_SIZE) pooled features (channels-first for consistency).
    """
    if x.dim() == 4:
        x = x.unsqueeze(0)
    # (B, T, H, W, 2) → (B, 2, T, H, W)
    x = x.permute(0, 4, 1, 2, 3)
    x = F.adaptive_avg_pool3d(x, FLOW_POOL_SIZE)
    return x


def pool_depth(x: torch.Tensor) -> torch.Tensor:
    """Adaptive-pool depth features to fixed size.

    Args:
        x: (B, T, H, W) or (T, H, W) raw depth features.
    Returns:
        (B, 1, *DEPTH_POOL_SIZE) pooled features.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    # (B, T, H, W) → (B, 1, T, H, W)
    x = x.unsqueeze(1)
    x = F.adaptive_avg_pool3d(x, DEPTH_POOL_SIZE)
    return x


def pool_track(x: torch.Tensor) -> torch.Tensor:
    """Adaptive-pool track features to fixed temporal size.

    Args:
        x: (B, T, 4) or (T, 4) raw track features.
    Returns:
        (B, TRACK_POOL_T, 4) pooled features.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    # (B, T, 4) → (B, 4, T) for pool1d → (B, 4, TRACK_POOL_T)
    x = x.permute(0, 2, 1)
    x = F.adaptive_avg_pool1d(x, TRACK_POOL_T)
    # → (B, TRACK_POOL_T, 4)
    x = x.permute(0, 2, 1)
    return x


def pool_dino(x: torch.Tensor) -> torch.Tensor:
    """Pass-through pool for DINO features.

    Dataset.py is expected to mean-collapse the temporal dim before calling
    this, so input is already (B, 768) or (768,).

    Args:
        x: (B, 768) or (768,) DINO mean-frame-diff embedding (z-score normalized).
    Returns:
        (B, 768) batched.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() != 2 or x.shape[-1] != DINO_FLAT:
        raise ValueError(
            f"pool_dino expects (B, {DINO_FLAT}) or ({DINO_FLAT},), got {tuple(x.shape)}. "
            f"Did dataset.py forget to mean-collapse the temporal dim?"
        )
    return x


def pool_freq(x: torch.Tensor) -> torch.Tensor:
    """Pass-through pool for FFT frequency energy features.

    Dataset.py is expected to mean-collapse the temporal dim before calling
    this, so input is a single scalar per sample.

    Args:
        x: (B, 1), (1,), or scalar FFT mean-frequency energy ([-1, 1] normalized).
    Returns:
        (B, 1) batched.
    """
    if x.dim() == 0:
        x = x.unsqueeze(0).unsqueeze(0)  # scalar → (1, 1)
    elif x.dim() == 1:
        x = x.unsqueeze(0) if x.shape[0] == FREQ_FLAT else x.unsqueeze(-1)
    if x.dim() != 2 or x.shape[-1] != FREQ_FLAT:
        raise ValueError(
            f"pool_freq expects (B, {FREQ_FLAT}), ({FREQ_FLAT},), or scalar, "
            f"got {tuple(x.shape)}. Did dataset.py forget to mean-collapse?"
        )
    return x


# Flat dimensions after pooling
FLOW_FLAT_DIM = FLOW_POOL_SIZE[0] * FLOW_POOL_SIZE[1] * FLOW_POOL_SIZE[2] * 2   # 512
DEPTH_FLAT_DIM = DEPTH_POOL_SIZE[0] * DEPTH_POOL_SIZE[1] * DEPTH_POOL_SIZE[2]   # 256
TRACK_FLAT_DIM = TRACK_POOL_T * 4                                                # 32
DINO_FLAT_DIM = DINO_FLAT                                                        # 768
FREQ_FLAT_DIM = FREQ_FLAT                                                        # 1

FLAT_DIMS = {
    "flow": FLOW_FLAT_DIM,
    "depth": DEPTH_FLAT_DIM,
    "track": TRACK_FLAT_DIM,
    "dino": DINO_FLAT_DIM,
    "freq": FREQ_FLAT_DIM,
}

POOL_FNS = {
    "flow": pool_flow,
    "depth": pool_depth,
    "track": pool_track,
    "dino": pool_dino,
    "freq": pool_freq,
}


class PhysicsTokenProjection(nn.Module):
    """Projects compressed physics features to token embeddings.

    Args:
        feature_type: One of "flow", "depth", "track", "dino", "freq".
        config: PhysicsTokenConfig (provides n_tokens and token_dim).
        hidden_dim: MLP intermediate dimension.
    """

    def __init__(
        self,
        feature_type: str,
        config: PhysicsTokenConfig,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        assert feature_type in ("flow", "depth", "track", "dino", "freq"), (
            f"Unknown feature_type: {feature_type}"
        )
        self.feature_type = feature_type
        self.n_tokens = getattr(config, f"{feature_type}_tokens")
        self.token_dim = config.token_dim
        self.pool_fn = POOL_FNS[feature_type]

        flat_dim = FLAT_DIMS[feature_type]

        # For freq (flat_dim=1), the standard MLP would be degenerate
        # (a single scalar → 14336 output). We still use the same architecture
        # but the first Linear becomes a scalar broadcast — sufficient for v8
        # since IBGate downstream will handle the actual token-level scoring.
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_tokens * self.token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw physics features (batched or unbatched).
               flow:  (B, T, H, W, 2) or (T, H, W, 2)
               depth: (B, T, H, W) or (T, H, W)
               track: (B, T, 4) or (T, 4)
               dino:  (B, 768) or (768,)        — already mean-collapsed
               freq:  (B, 1), (1,), or scalar   — already mean-collapsed

        Returns:
            Token embeddings of shape (B, n_tokens, token_dim).
        """
        pooled = self.pool_fn(x)           # (B, ...) fixed-size
        B = pooled.shape[0]
        flat = pooled.reshape(B, -1)       # (B, flat_dim)
        out = self.mlp(flat)               # (B, n_tokens * token_dim)
        return out.view(B, self.n_tokens, self.token_dim)


class PhysicsProjectionBundle(nn.Module):
    """Bundle of all five physics token projections.

    Convenience wrapper that holds flow/depth/track/dino/freq projections and
    provides a single forward call returning all token embeddings.

    For v5 backward-compat, set config.dino_tokens=0 and config.freq_tokens=0
    and call forward() without dino_feat/freq_feat (they default to None).
    """

    def __init__(self, config: PhysicsTokenConfig, hidden_dim: int = 1024):
        super().__init__()
        self.config = config
        self.flow_proj = PhysicsTokenProjection("flow", config, hidden_dim)
        self.depth_proj = PhysicsTokenProjection("depth", config, hidden_dim)
        self.track_proj = PhysicsTokenProjection("track", config, hidden_dim)
        # dino/freq are optional — only instantiate if non-zero token count
        self.has_dino = config.dino_tokens > 0
        self.has_freq = config.freq_tokens > 0
        if self.has_dino:
            self.dino_proj = PhysicsTokenProjection("dino", config, hidden_dim)
        if self.has_freq:
            self.freq_proj = PhysicsTokenProjection("freq", config, hidden_dim)

    def forward(
        self,
        flow_feat,
        depth_feat,
        track_feat,
        dino_feat=None,
        freq_feat=None,
    ):
        """
        Args:
            flow_feat / depth_feat / track_feat: required raw physics features
            dino_feat:  (B, 768) or None — required iff config.dino_tokens > 0
            freq_feat:  (B, 1)   or None — required iff config.freq_tokens > 0

        Returns:
            Dict with keys "flow", "depth", "track", and (if enabled)
            "dino", "freq", each (B, n_tokens, token_dim).
        """
        out = {
            "flow": self.flow_proj(flow_feat),
            "depth": self.depth_proj(depth_feat),
            "track": self.track_proj(track_feat),
        }
        if self.has_dino:
            if dino_feat is None:
                raise ValueError("dino_tokens > 0 but dino_feat is None")
            out["dino"] = self.dino_proj(dino_feat)
        if self.has_freq:
            if freq_feat is None:
                raise ValueError("freq_tokens > 0 but freq_feat is None")
            out["freq"] = self.freq_proj(freq_feat)
        return out
