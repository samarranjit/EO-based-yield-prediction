"""Prithvi-EO-2.0 backbone adapter.

Wraps the official Prithvi-EO-2.0-600M(-TL) encoder behind a clean interface:

* accepts ``x = [B, 6, T, 224, 224]`` plus optional temporal/location metadata,
* runs the encoder's ``forward_features`` (variable-T is handled by Prithvi's
  own ``interpolate_pos_encoding``),
* selects the paper's transformer blocks {8,16,24,32},
* returns each as a 2-D-ready feature ``[B, D, T, gh, gw]``,
* supports frozen / partially-frozen / full fine-tuning and gradient checkpointing.

The real backbone is loaded through **TerraTorch** (optional extra
``[prithvi]``). For unit tests and CPU smoke tests we provide
:class:`DummyPrithviBackbone`, a tiny transformer with the *same* token layout
(CLS + T·gh·gw tokens, embed dim configurable) so every downstream module is
exercised without downloading 600M weights.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from ..utils.logging import FarmError, get_logger
from .feature_extractor import one_based_to_index, tokens_to_feature_map

logger = get_logger(__name__)


class BackboneNotAvailable(FarmError):
    pass


class FeatureBackbone(Protocol):
    embed_dim: int
    depth: int

    def extract_features(
        self,
        x: torch.Tensor,
        temporal_coords: torch.Tensor | None,
        location_coords: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        ...


# --------------------------------------------------------------------------- #
# Dummy backbone (test / CPU smoke)
# --------------------------------------------------------------------------- #

class DummyPrithviBackbone(nn.Module):
    """Lightweight stand-in with Prithvi's token contract.

    Produces ``depth`` per-block token tensors ``[B, 1 + T*gh*gw, embed_dim]``
    from ``[B, C, T, H, W]`` input, so feature selection / reshape / decoder can
    be tested end-to-end on CPU.
    """

    def __init__(
        self,
        in_chans: int = 6,
        embed_dim: int = 64,
        depth: int = 32,
        patch_size: int = 14,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim, kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward=embed_dim * 2,
            batch_first=True, activation="gelu",
        )
        self.blocks = nn.ModuleList([layer for _ in range(depth)])
        # independent params per block:
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    embed_dim, num_heads, dim_feedforward=embed_dim * 2,
                    batch_first=True, activation="gelu",
                )
                for _ in range(depth)
            ]
        )

    def forward_features(
        self,
        x: torch.Tensor,
        temporal_coords: torch.Tensor | None = None,
        location_coords: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        b = x.shape[0]
        z = self.patch_embed(x)  # [B, D, T, gh, gw]
        z = z.flatten(2).transpose(1, 2)  # [B, T*gh*gw, D]  (order t,h,w preserved)
        cls = self.cls_token.expand(b, -1, -1)
        z = torch.cat([cls, z], dim=1)
        outs: list[torch.Tensor] = []
        for blk in self.blocks:
            z = blk(z)
            outs.append(z)
        return outs


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #

class PrithviAdapter(nn.Module):
    def __init__(
        self,
        backbone_id: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
        pretrained: bool = True,
        in_chans: int = 6,
        n_timesteps: int = 8,
        out_blocks_one_based: list[int] | None = None,
        finetune_mode: str = "full",
        gradient_checkpointing: bool = False,
        use_dummy: bool = False,
        dummy_embed_dim: int = 64,
        expected_embed_dim: int = 1280,
        depth: int = 32,
    ) -> None:
        super().__init__()
        self.n_timesteps = n_timesteps
        self.out_blocks_one_based = out_blocks_one_based or [8, 16, 24, 32]
        self.finetune_mode = finetune_mode
        self.gradient_checkpointing = gradient_checkpointing
        self.use_dummy = use_dummy

        if use_dummy:
            self.backbone: nn.Module = DummyPrithviBackbone(
                in_chans=in_chans, embed_dim=dummy_embed_dim, depth=depth
            )
            self.embed_dim = dummy_embed_dim
            self.depth = depth
        else:
            self.backbone = self._load_terratorch(backbone_id, pretrained, in_chans, n_timesteps)
            self.embed_dim = getattr(self.backbone, "embed_dim", expected_embed_dim)
            self.depth = getattr(self.backbone, "depth", depth) or depth

        self.out_indices = one_based_to_index(self.out_blocks_one_based, self.depth)
        self._apply_finetune_mode()

    # --- loading --- #
    @staticmethod
    def _load_terratorch(backbone_id: str, pretrained: bool, in_chans: int, n_frames: int):
        """Load the real Prithvi encoder via TerraTorch's backbone registry.

        We keep this behind a clear error so the rest of the package runs without
        the heavy ``[prithvi]`` extra installed.
        """
        try:
            from terratorch.registry import BACKBONE_REGISTRY
        except Exception as e:  # pragma: no cover - needs optional extra
            raise BackboneNotAvailable(
                "TerraTorch is required for the real Prithvi backbone. "
                "Install with `uv sync --extra prithvi`, or pass use_dummy=True.\n"
                f"Underlying import error: {e}"
            ) from e

        # Map HF id → terratorch backbone key.
        key = "prithvi_eo_v2_600_tl" if backbone_id.endswith("TL") else "prithvi_eo_v2_600"
        logger.info("Building TerraTorch backbone '%s' (pretrained=%s)", key, pretrained)
        model = BACKBONE_REGISTRY.build(
            key,
            pretrained=pretrained,
            num_frames=n_frames,
            bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        )
        return model

    # --- freezing --- #
    def _apply_finetune_mode(self) -> None:
        if self.finetune_mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad_(True)
        elif self.finetune_mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        elif self.finetune_mode == "decoder_only":
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        else:
            raise ValueError(f"Unknown finetune_mode {self.finetune_mode!r}")

    def trainable_parameter_count(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    # --- forward --- #
    def _forward_features(self, x, temporal_coords, location_coords):
        fn = self.backbone.forward_features
        if self.gradient_checkpointing and self.training:
            import torch.utils.checkpoint as ckpt

            return ckpt.checkpoint(
                lambda a: fn(a, temporal_coords, location_coords), x, use_reentrant=False
            )
        return fn(x, temporal_coords, location_coords)

    def extract_features(
        self,
        x: torch.Tensor,
        temporal_coords: torch.Tensor | None = None,
        location_coords: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Return selected-block features as ``[B, D, T, gh, gw]`` each."""
        if x.dim() != 5:
            raise ValueError(f"Expected x [B,C,T,H,W], got shape {tuple(x.shape)}")
        all_tokens = self._forward_features(x, temporal_coords, location_coords)
        if len(all_tokens) < self.depth:
            # Some implementations return depth+1 (with final norm); index still valid.
            pass
        feats: list[torch.Tensor] = []
        for i in self.out_indices:
            tok = all_tokens[i]
            feats.append(tokens_to_feature_map(tok, self.n_timesteps, has_cls_token=True))
        return feats

    def forward(self, x, temporal_coords=None, location_coords=None):
        return self.extract_features(x, temporal_coords, location_coords)
