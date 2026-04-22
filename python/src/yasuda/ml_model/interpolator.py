from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = getLogger()


class Interpolator(nn.Module):
    def __init__(
        self,
        *,
        interpolation_mode: str,
        align_corners: bool,
        scale: int,
        **kwargs,
    ):
        super().__init__()

        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.scale = scale

        logger.info(f"scale = {self.scale}")
        logger.info(f"interpolation mode = {self.interpolation_mode}")
        logger.info(f"align corners = {self.align_corners}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.interpolation_mode == "nearest":
            return F.interpolate(x, scale_factor=self.scale, mode="nearest")
        else:
            return F.interpolate(
                x,
                scale_factor=self.scale,
                mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )