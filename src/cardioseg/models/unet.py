"""A U-Net: the encoder works out WHAT is in the image, the decoder works out WHERE.

Shapes for one 224x224 slice with base_channels=32:
    input      (B, 1, 224, 224)
    encoder    32@224 -> 64@112 -> 128@56 -> 256@28 -> bottom 512@14
    decoder    256@28 -> 128@56 -> 64@112 -> 32@224
    output     (B, 4, 224, 224)  one score per class (background, RV, myocardium, LV) per pixel

Options (each one becomes a row in the ablation table):
    residual=True   use ResidualBlock instead of ConvBlock everywhere
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two rounds of: 3x3 convolution -> batch norm -> ReLU. Keeps height and width the same."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualBlock(nn.Module):
    """Same job as ConvBlock, plus a shortcut: output = ReLU(F(x) + x).

    The convolutions only have to learn what to ADD to x (the "residual"), and the
    shortcut gives the training signal a direct path back to earlier layers.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # x can only be added to F(x) if they have the same number of channels.
        # If not, a 1x1 convolution changes x's channel count first.
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.layers(x) + self.shortcut(x))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        residual: bool = False,
    ):
        super().__init__()
        c = base_channels
        Block = ResidualBlock if residual else ConvBlock

        # Encoder (left side of the U): channels double as the image halves.
        self.enc1 = Block(in_channels, c)
        self.enc2 = Block(c, c * 2)
        self.enc3 = Block(c * 2, c * 4)
        self.enc4 = Block(c * 4, c * 8)
        self.bottom = Block(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)

        # Decoder (right side): upsample x2, glue on the matching encoder map, convolve.
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = Block(c * 16, c * 8)  # c*8 from below + c*8 from the skip
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = Block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = Block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = Block(c * 2, c)

        # 1x1 convolution: turns c feature maps into one score map per class.
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottom(self.pool(e4))

        # torch.cat(..., dim=1) stacks along the channel dimension: the skip connection.
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
