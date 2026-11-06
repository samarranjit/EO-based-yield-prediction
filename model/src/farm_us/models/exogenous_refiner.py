"""ExogenousRefiner: the sub-token residual branch, reading raw imagery plus
soil (POLARIS, 30 m, static) and weather (PRISM, monthly, coarse) covariates.

    main = coarse + ExogenousRefiner(image, soil, weather, body)

This SUPERSEDES ``detail_refiner.py`` rather than stacking on it. The raw-imagery
path here is DetailRefiner's, unchanged in structure: the same reshape of
``[B,C,T,H,W]`` to ``[B,C*T,H,W]``, the same two-conv ``pix`` stack, the same
fuse against the head's body features, the same zero-init ``out``. Soil is
concatenated into that stack and weather modulates it. Set ``n_soil=0`` and
``n_weather=0`` and this module IS DetailRefiner, which makes that configuration
the clean image-only ablation for measuring what the covariates add.

``cfg.detail_refiner`` and ``cfg.exogenous_refiner`` are therefore mutually
exclusive -- ``FarmModel.__init__`` raises if both are set, because two
image-derived residuals on one coarse prediction double-count the sub-token
signal and split its gradient across branches that cannot see each other.

``detail_refiner.py`` is kept byte-for-byte so the BARC ``refiner_only`` runs
(``barc_refiner*.yaml``, mean pixel pearson_r2 0.465) stay reproducible from
their checkpoints; that is its only remaining role. New work -- whether trained
from scratch, from ``init_from`` on a checkpoint without any branch (farm-018),
or alone against a frozen base via ``finetune_mode=exo_refiner_only`` -- uses
this module. See models/farm_model.py.

Spatial covariates
------------------
Soil enters the spatial CNN at the chip resolution. Monthly weather is read
with nearest-neighbor reprojection: each target pixel retains the value of its
source PRISM cell. No chip averaging or bilinear weather interpolation occurs.
A pointwise (1x1 convolution) weather network generates scale/shift MAPS, so
conditioning varies across PRISM cells without inventing finer observations.

Zero-init contract (identical to DetailRefiner)
------------------------------------------------
``out`` starts at zero weight and bias, and the FiLM head's ``to_scale`` /
``to_shift`` layers also start at zero, so at step 0:
  - the spatial branch is unaffected by soil channels having arbitrary values
    (they still flow through ``pix``, but ``out`` maps everything to exactly 0)
  - weather conditioning is the identity (scale=0 -> 1+scale=1, shift=0)
so the branch contributes exactly nothing and reproduces the source checkpoint
bit-for-bit under ``init_from``. Subsequent learned corrections can improve or worsen predictions.

No BatchNorm, for the same reason as DetailRefiner: fine-tuning batches are
small and a frozen base (``exo_refiner_only``) cannot afford batch statistics
that drift.

Either covariate is independently optional: ``n_soil=0`` skips the soil concat
(falls back to image-only spatial input, like DetailRefiner); ``n_weather=0``
skips the FiLM branch entirely (no conditioning module is even built).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherEncoder(nn.Module):
    """Pointwise weather network: [B,W,T,H,W] -> spatial FiLM maps.

    Flatten variable/month axes only. Each pixel uses the same learned mapping
    of its monthly weather sequence; 1x1 convolutions never mix nearby cells.
    """

    def __init__(self, n_weather: int, n_timesteps: int, hidden: int) -> None:
        super().__init__()
        self.n_weather, self.n_timesteps = n_weather, n_timesteps
        in_dim = n_weather * n_timesteps
        self.net = nn.Sequential(nn.Conv2d(in_dim, hidden, 1), nn.ReLU(inplace=True))
        self.to_scale = nn.Conv2d(hidden, hidden, 1)
        self.to_shift = nn.Conv2d(hidden, hidden, 1)
        # Zero-init: at step 0, scale=0 (so 1+scale=1) and shift=0 -- FiLM starts
        # as the identity transform, matching the branch's overall zero-init
        # contract. nn.init.zeros_ on BOTH weight and bias is required: a
        # zero-bias-only linear layer still outputs nonzero values from
        # nonzero inputs through its weight matrix.
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, weather: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if weather.ndim != 5 or weather.shape[1:3] != (self.n_weather, self.n_timesteps):
            raise ValueError(
                "weather must have shape [B, n_weather, n_timesteps, H, W]; "
                f"expected W={self.n_weather}, T={self.n_timesteps}, got {tuple(weather.shape)}. "
                "Legacy chip-mean weather inputs/checkpoints are not compatible."
            )
        b, _, _, height, width = weather.shape
        h = self.net(weather.reshape(b, -1, height, width))
        return self.to_scale(h), self.to_shift(h)  # each [B, hidden, H, W]


class ExogenousRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,       # image channels = bands * timesteps
        feat_channels: int,     # head body channels (64)
        n_soil: int = 0,
        n_weather: int = 0,
        n_timesteps: int = 8,
        hidden: int = 16,
    ) -> None:
        super().__init__()
        self.n_soil = n_soil
        self.n_weather = n_weather

        pix_in = in_channels + n_soil
        self.pix = nn.Sequential(
            nn.Conv2d(pix_in, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.weather_enc = (
            WeatherEncoder(n_weather, n_timesteps, hidden) if n_weather > 0 else None
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden + feat_channels, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, 1, kernel_size=1)
        assert self.out.bias is not None  # Conv2d default bias=True; narrows the type
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        image: torch.Tensor,
        body: torch.Tensor,
        soil: torch.Tensor | None = None,
        weather: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # image [B,C,T,H,W] -> stack time into channels for THIS branch only,
        # exactly as DetailRefiner does. The encoder still receives the 5-D
        # tensor unchanged elsewhere.
        b, c, t, h, w = image.shape
        img = image.reshape(b, c * t, h, w)

        if self.n_soil > 0:
            if soil is None:
                raise ValueError(
                    "ExogenousRefiner configured with n_soil>0 but soil=None was passed"
                )
            img = torch.cat([img, soil], dim=1)

        h_feat = self.pix(img)

        if self.weather_enc is not None:
            if weather is None:
                raise ValueError(
                    "ExogenousRefiner configured with n_weather>0 but weather=None was passed"
                )
            if weather.shape[0] != b or weather.shape[-2:] != (h, w):
                raise ValueError("weather batch and spatial grid must match image [B,C,T,H,W]")
            scale, shift = self.weather_enc(weather)
            h_feat = h_feat * (1.0 + scale) + shift

        feat = F.interpolate(body, size=(h, w), mode="bilinear", align_corners=False)
        return self.out(self.fuse(torch.cat([h_feat, feat], dim=1)))
