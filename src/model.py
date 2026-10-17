"""
3-D Residual U-Net for brain tumour segmentation.

Architecture
────────────
• Encoder: 4 levels, each = ResBlock(s) + MaxPool3d
• Bottleneck: ResBlock
• Decoder: 4 levels, each = Upsample + concat skip + ResBlock
• Head: 1×1×1 conv → num_classes logits

Filters at each level: base_filters × [1, 2, 4, 8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Two ConvBnRelu layers with a residual shortcut."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1  = ConvBnRelu(in_ch,  out_ch)
        self.conv2  = ConvBnRelu(out_ch, out_ch)
        self.drop   = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
        # Projection shortcut when channel dim changes
        self.shortcut = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        return F.relu(out + residual, inplace=True)


class EncoderBlock(nn.Module):
    """ResBlock(s) → store skip → MaxPool3d."""

    def __init__(self, in_ch: int, out_ch: int,
                 depth: int = 1, dropout: float = 0.0):
        super().__init__()
        blocks = [ResBlock(in_ch, out_ch, dropout)]
        for _ in range(depth - 1):
            blocks.append(ResBlock(out_ch, out_ch, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.pool   = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.blocks(x)
        down = self.pool(skip)
        return down, skip


class DecoderBlock(nn.Module):
    """Trilinear upsample → concat skip → ResBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.0):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode="trilinear",
                                 align_corners=True)
        self.block = ResBlock(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Crop skip to match x spatial dims (handles odd sizes)
        x = _crop_to(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


def _crop_to(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Centre-crop *src* to the spatial size of *ref*."""
    _, _, d, h, w = ref.shape
    _, _, sd, sh, sw = src.shape
    zs = (sd - d) // 2
    ys = (sh - h) // 2
    xs = (sw - w) // 2
    return src[:, :, zs:zs+d, ys:ys+h, xs:xs+w]


# ─────────────────────────────────────────────────────────────────────────────
# Full 3-D U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    Parameters
    ----------
    in_channels   : number of input modalities (4 for BraTS)
    num_classes   : segmentation classes (4 for BraTS remapped)
    base_filters  : feature maps at first encoder level (doubled each level)
    encoder_depths: list of ResBlock repetitions per encoder level
    dropout       : dropout probability inside ResBlocks
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
        encoder_depths: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if encoder_depths is None:
            encoder_depths = [1, 2, 2, 2]

        f = base_filters
        ch = [f, f*2, f*4, f*8]   # channels at each level

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels, ch[0], encoder_depths[0], dropout)
        self.enc2 = EncoderBlock(ch[0],       ch[1], encoder_depths[1], dropout)
        self.enc3 = EncoderBlock(ch[1],       ch[2], encoder_depths[2], dropout)
        self.enc4 = EncoderBlock(ch[2],       ch[3], encoder_depths[3], dropout)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ResBlock(ch[3], ch[3] * 2, dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(ch[3]*2, ch[3], ch[3], dropout)
        self.dec3 = DecoderBlock(ch[3],   ch[2], ch[2], dropout)
        self.dec2 = DecoderBlock(ch[2],   ch[1], ch[1], dropout)
        self.dec1 = DecoderBlock(ch[1],   ch[0], ch[0], dropout)

        # ── Segmentation head ─────────────────────────────────────────────────
        self.head = nn.Conv3d(ch[0], num_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, D, H, W)

        Returns
        -------
        logits : (B, num_classes, D, H, W)
        """
        # Encoder
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def count_parameters(model: "UNet3D") -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet3D(in_channels=4, num_classes=4, base_filters=32).to(device)
    x      = torch.randn(1, 4, 128, 128, 128, device=device)
    logits = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {logits.shape}")
    print(f"Params: {UNet3D.count_parameters(model):,}")
