"""Decoder: maps a 512-d ResNet-18 embedding back to a 32x32 RGB image."""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """512-d embedding -> 32x32x3 RGB image in [0, 1].

    FC projects to 4x4x256, then three ConvTranspose stages double the spatial
    size: 4 -> 8 -> 16 -> 32. Final 1x1 conv + sigmoid yields 3-channel pixels.
    """

    def __init__(self, embed_dim: int = 512, base_ch: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.base_ch = base_ch

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, base_ch * 4 * 4),
            nn.BatchNorm1d(base_ch * 4 * 4),
            nn.ReLU(inplace=True),
        )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2,
                                   padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.up1 = up_block(base_ch, base_ch // 2)       # 4 -> 8
        self.up2 = up_block(base_ch // 2, base_ch // 4)  # 8 -> 16
        self.up3 = up_block(base_ch // 4, base_ch // 8)  # 16 -> 32
        self.to_rgb = nn.Conv2d(base_ch // 8, 3, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.base_ch, 4, 4)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        return torch.sigmoid(self.to_rgb(h))
