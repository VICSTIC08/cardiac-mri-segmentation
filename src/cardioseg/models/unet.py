"""A plain U-Net: the encoder works out WHAT is in the image, the decoder works out WHERE.

Shapes for one 224x224 slice with base_channels=32:
    input      (B, 1, 224, 224)
    encoder    32@224 -> 64@112 -> 128@56 -> 256@28 -> bottom 512@14
    decoder    256@28 -> 128@56 -> 64@112 -> 32@224
    output     (B, 4, 224, 224)  one score per class (background, RV, myocardium, LV) per pixel
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


class UNet(nn.Module):
    def __init__(
        self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32
    ):
        super().__init__()
        c = base_channels
        # Encoder (left side of the U): channels double as the image halves.
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bottom = ConvBlock(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)

        # Decoder (right side): upsample x2, glue on the matching encoder map, convolve.
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)  # c*8 from below + c*8 from the skip
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

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
