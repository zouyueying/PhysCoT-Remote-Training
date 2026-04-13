"""Lightweight decoders: token hidden states → reconstructed physics features.

Used in Stage 1 to compute reconstruction loss. Each decoder maps the LLM's
hidden states at physics token positions back to the compressed (pooled)
physics feature space.

Data flow (Stage 1 reconstruction):
    GT feature → pool → compressed GT  ─────────────────── ┐
    GT feature → projection → token embeddings → LLM      │
         LLM hidden states at token positions → decoder →  MSE loss
                                                 ↑         │
                                        reconstructed  ←───┘

v8 (5-type): adds dino_dec and freq_dec.
- dino: hidden → (768,) reconstructed mean-pooled DINO embedding
- freq: hidden → (1,)   reconstructed scaled FFT energy scalar
"""

import torch
import torch.nn as nn

from physcot.tokens.physics_token import PhysicsTokenConfig
from physcot.tokens.projection import (
    DEPTH_FLAT_DIM,
    DEPTH_POOL_SIZE,
    DINO_FLAT_DIM,
    FLOW_FLAT_DIM,
    FLOW_POOL_SIZE,
    FREQ_FLAT_DIM,
    TRACK_FLAT_DIM,
    TRACK_POOL_T,
)


class PhysicsFeatureDecoder(nn.Module):
    """Decodes LLM hidden states back to compressed physics features.

    Args:
        feature_type: One of "flow", "depth", "track", "dino", "freq".
        config: PhysicsTokenConfig.
        hidden_dim: MLP intermediate dimension.
    """

    def __init__(
        self,
        feature_type: str,
        config: PhysicsTokenConfig,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        assert feature_type in ("flow", "depth", "track", "dino", "freq")
        self.feature_type = feature_type
        self.n_tokens = getattr(config, f"{feature_type}_tokens")
        self.token_dim = config.token_dim

        if feature_type == "flow":
            self.flat_dim = FLOW_FLAT_DIM
            self.output_shape = (2, *FLOW_POOL_SIZE)  # (2, T, H, W)
        elif feature_type == "depth":
            self.flat_dim = DEPTH_FLAT_DIM
            self.output_shape = (1, *DEPTH_POOL_SIZE)  # (1, T, H, W)
        elif feature_type == "track":
            self.flat_dim = TRACK_FLAT_DIM
            self.output_shape = (TRACK_POOL_T, 4)      # (T, 4)
        elif feature_type == "dino":
            self.flat_dim = DINO_FLAT_DIM              # 768
            self.output_shape = (DINO_FLAT_DIM,)        # (768,) — flat vector
        else:  # freq
            self.flat_dim = FREQ_FLAT_DIM              # 1
            self.output_shape = (FREQ_FLAT_DIM,)        # (1,) — scalar

        self.mlp = nn.Sequential(
            nn.Linear(self.n_tokens * self.token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.flat_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, n_tokens, token_dim) — LLM hidden states at
                physics token positions.

        Returns:
            Reconstructed compressed features:
                flow:  (B, 2, T_pool, H_pool, W_pool)
                depth: (B, 1, T_pool, H_pool, W_pool)
                track: (B, T_pool, 4)
                dino:  (B, 768)
                freq:  (B, 1)
        """
        B = hidden_states.shape[0]
        flat = hidden_states.reshape(B, -1)          # (B, n_tokens * token_dim)
        out = self.mlp(flat)                          # (B, flat_dim)
        return out.view(B, *self.output_shape)


class PhysicsDecoderBundle(nn.Module):
    """Bundle of all five physics feature decoders.

    For v5 backward-compat, set config.dino_tokens=0 and config.freq_tokens=0
    so dino_dec and freq_dec are not instantiated.
    """

    def __init__(self, config: PhysicsTokenConfig, hidden_dim: int = 1024):
        super().__init__()
        self.config = config
        self.flow_dec = PhysicsFeatureDecoder("flow", config, hidden_dim)
        self.depth_dec = PhysicsFeatureDecoder("depth", config, hidden_dim)
        self.track_dec = PhysicsFeatureDecoder("track", config, hidden_dim)
        self.has_dino = config.dino_tokens > 0
        self.has_freq = config.freq_tokens > 0
        if self.has_dino:
            self.dino_dec = PhysicsFeatureDecoder("dino", config, hidden_dim)
        if self.has_freq:
            self.freq_dec = PhysicsFeatureDecoder("freq", config, hidden_dim)

    def forward(
        self,
        flow_hidden,
        depth_hidden,
        track_hidden,
        dino_hidden=None,
        freq_hidden=None,
    ):
        """
        Args:
            *_hidden: (B, n_tokens, token_dim) hidden states per feature type.
                      dino_hidden / freq_hidden are optional (v5 → None).

        Returns:
            Dict with keys for the active types.
        """
        out = {
            "flow": self.flow_dec(flow_hidden),
            "depth": self.depth_dec(depth_hidden),
            "track": self.track_dec(track_hidden),
        }
        if self.has_dino and dino_hidden is not None:
            out["dino"] = self.dino_dec(dino_hidden)
        if self.has_freq and freq_hidden is not None:
            out["freq"] = self.freq_dec(freq_hidden)
        return out

    def reconstruction_loss(
        self,
        flow_hidden, depth_hidden, track_hidden,
        flow_gt_pooled, depth_gt_pooled, track_gt_pooled,
        dino_hidden=None, freq_hidden=None,
        dino_gt=None, freq_gt=None,
        lambda_flow: float = 1.0,
        lambda_depth: float = 1.0,
        lambda_track: float = 1.0,
        lambda_dino: float = 0.5,
        lambda_freq: float = 0.5,
    ) -> dict:
        """Compute weighted reconstruction MSE loss across active types.

        v8: dino/freq are skipped if their hidden/gt is None (lenient mode for
        samples missing those features).

        Returns:
            Dict with "loss" (scalar), per-type losses, and dict of which
            types contributed (for logging).
        """
        preds = self.forward(
            flow_hidden, depth_hidden, track_hidden,
            dino_hidden=dino_hidden, freq_hidden=freq_hidden,
        )
        mse = nn.functional.mse_loss

        flow_loss = mse(preds["flow"], flow_gt_pooled)
        depth_loss = mse(preds["depth"], depth_gt_pooled)
        track_loss = mse(preds["track"], track_gt_pooled)

        total = (
            lambda_flow * flow_loss
            + lambda_depth * depth_loss
            + lambda_track * track_loss
        )

        result = {
            "flow_loss": flow_loss,
            "depth_loss": depth_loss,
            "track_loss": track_loss,
            "dino_loss": 0.0,
            "freq_loss": 0.0,
        }

        if self.has_dino and dino_hidden is not None and dino_gt is not None:
            dino_loss = mse(preds["dino"], dino_gt)
            total = total + lambda_dino * dino_loss
            result["dino_loss"] = dino_loss

        if self.has_freq and freq_hidden is not None and freq_gt is not None:
            freq_loss = mse(preds["freq"], freq_gt)
            total = total + lambda_freq * freq_loss
            result["freq_loss"] = freq_loss

        result["loss"] = total
        return result
