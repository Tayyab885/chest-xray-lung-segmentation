"""U-Net variants. Both return logits, never probabilities."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


def _check_divisible(x, divisor):
    """Reject input sizes the skip connections cannot line up again.

    Encoder pooling floors, decoder upsampling doubles exactly, so an odd size
    anywhere makes a skip arrive one pixel off and torch.cat raises deep inside
    forward. Training runs at 512, but a stray crop would otherwise surface as a
    confusing shape error rather than a statement of the constraint.
    """
    h, w = x.shape[-2:]
    if h % divisor or w % divisor:
        raise ValueError(
            f"input {h}x{w} must be divisible by {divisor} in both dimensions"
        )


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Classic U-Net, trained from scratch."""

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList()
        prev = in_ch
        for w in widths:
            self.enc.append(conv_block(prev, w))
            prev = w
        self.bottleneck = conv_block(widths[-1], widths[-1] * 2)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = widths[-1] * 2
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(prev, w, 2, stride=2))
            self.dec.append(conv_block(w * 2, w))
            prev = w
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        _check_divisible(x, 16)  # four pooling stages
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


class UNetResNet34(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet34 encoder.

    The single-channel radiograph is repeated to three channels so the
    pretrained stem can be used unmodified.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet34(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)  # /2, 64ch
        self.pool = net.maxpool
        self.layer1 = net.layer1   # /4,  64ch
        self.layer2 = net.layer2   # /8,  128ch
        self.layer3 = net.layer3   # /16, 256ch
        self.layer4 = net.layer4   # /32, 512ch

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = conv_block(32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        _check_divisible(x, 32)  # ResNet34 downsamples 32x
        x = x.repeat(1, 3, 1, 1)
        s0 = self.stem(x)          # /2
        s1 = self.layer1(self.pool(s0))  # /4
        s2 = self.layer2(s1)       # /8
        s3 = self.layer3(s2)       # /16
        s4 = self.layer4(s3)       # /32

        d4 = self.dec4(torch.cat([self.up4(s4), s3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s0], dim=1))
        d0 = self.dec0(self.up0(d1))
        return self.head(d0)


def build_model(name, pretrained=True):
    if name == "unet":
        if pretrained:
            raise ValueError(
                "the scratch UNet has no pretrained variant, pass pretrained=False. "
                "Silently ignoring the flag would let a run matrix produce two "
                "identical scratch runs and read them as a comparison."
            )
        return UNet()
    if name == "unet_resnet34":
        return UNetResNet34(pretrained=pretrained)
    raise ValueError(f"unknown model name: {name!r}, expected 'unet' or 'unet_resnet34'")
