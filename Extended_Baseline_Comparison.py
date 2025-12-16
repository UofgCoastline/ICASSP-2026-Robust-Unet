#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Baseline Comparison for Coastal Water Segmentation
Includes water-specific segmentation methods and comprehensive visualization

Baselines included:
1. Robust U-Net (Ours)
2. DeepLabV3+ (General SOTA)
3. YOLO-SEG (General SOTA)
4. WaterNet - Water-specific segmentation network
5. MSWNet - Multi-Scale Water Network
6. HRNet-Water - High-Resolution Water Segmentation
7. SegFormer - Transformer-based segmentation

This script addresses baseline comparison comprehensiveness.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import json
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# Dataset
# =============================================================================
class CoastalDataset(Dataset):
    """Coastal water segmentation dataset with LabelMe annotations"""

    def __init__(self, image_paths, label_paths, transform=None, image_size=(512, 512)):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.load_image(self.image_paths[idx])
        mask = self.create_mask_from_labelme(self.label_paths[idx], image.size)

        image = image.resize(self.image_size, Image.LANCZOS)
        mask = Image.fromarray(mask).resize(self.image_size, Image.NEAREST)
        mask = np.array(mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return image, mask, self.image_paths[idx]

    def load_image(self, image_path):
        try:
            return Image.open(image_path).convert('RGB')
        except:
            return Image.new('RGB', (512, 512), (128, 128, 128))

    def create_mask_from_labelme(self, label_path, image_size):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)

            mask_image = Image.new('L', image_size, 0)
            draw = ImageDraw.Draw(mask_image)

            for shape in label_data.get('shapes', []):
                if shape['label'].lower() in ['water', 'sea', '海水', '水体']:
                    points = [(int(p[0]), int(p[1])) for p in shape['points']]
                    if len(points) >= 3:
                        draw.polygon(points, fill=1)

            return np.array(mask_image, dtype=np.uint8)
        except:
            return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)


# =============================================================================
# Attention Modules (Shared)
# =============================================================================
class ChannelAttention(nn.Module):
    """Channel Attention Module (CBAM)"""

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (CBAM)"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(x_att))


class AttentionGate(nn.Module):
    """Attention Gate for Skip Connections"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# =============================================================================
# Model 1: Robust U-Net (Ours)
# =============================================================================
class ResidualBlock(nn.Module):
    """Residual Block with attention"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        out = self.sa(out)
        return self.relu(out + residual)


class DilatedBlock(nn.Module):
    """Multi-scale dilated convolution block"""

    def __init__(self, in_channels, out_channels):
        super(DilatedBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=4, dilation=4)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return self.relu(self.bn(torch.cat([x1, x2, x3, x4], dim=1)))


class RobustUNet(nn.Module):
    """Robust U-Net with attention mechanisms (Our Method)"""

    def __init__(self, n_channels=3, n_classes=1, base_channels=64):
        super(RobustUNet, self).__init__()

        # Encoder
        self.inc = ResidualBlock(n_channels, base_channels, dropout_rate=0.1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels, base_channels * 2, 0.1))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels * 2, base_channels * 4, 0.2))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels * 4, base_channels * 8, 0.2))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DilatedBlock(base_channels * 8, base_channels * 16),
            ResidualBlock(base_channels * 16, base_channels * 16, 0.3)
        )

        # Attention gates
        self.att4 = AttentionGate(base_channels * 8, base_channels * 8, base_channels * 4)
        self.att3 = AttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
        self.att2 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.att1 = AttentionGate(base_channels, base_channels, base_channels // 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = ResidualBlock(base_channels * 16, base_channels * 8, 0.2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4, 0.2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2, 0.1)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, 0.1)

        self.outc = nn.Sequential(nn.Conv2d(base_channels, n_classes, 1), nn.Sigmoid())
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up4(x5)
        x = torch.cat([self.att4(x, x4), x], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([self.att3(x, x3), x], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([self.att2(x, x2), x], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([self.att1(x, x1), x], dim=1)
        x = self.dec1(x)

        return self.outc(x)


# =============================================================================
# Model 2: DeepLabV3+
# =============================================================================
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        size = x.shape[-2:]
        x1, x2, x3, x4 = self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)
        x5 = F.interpolate(self.conv5(self.global_pool(x)), size=size, mode='bilinear', align_corners=False)
        return F.relu(self.bn(self.conv_out(torch.cat([x1, x2, x3, x4, x5], dim=1))))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ for semantic segmentation"""

    def __init__(self, n_classes=1):
        super(DeepLabV3Plus, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.MaxPool2d(3, stride=2, padding=1), nn.Conv2d(64, 128, 3, padding=1),
                                   nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True))
        self.aspp = ASPP(512, 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 3, padding=1)
        )

    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        return torch.sigmoid(self.decoder(self.aspp(x)))


# =============================================================================
# Model 3: YOLO-SEG
# =============================================================================
class YOLOSeg(nn.Module):
    """YOLO-style segmentation network"""

    def __init__(self, n_classes=1):
        super(YOLOSeg, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, n_classes, 3, padding=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.seg_head(self.backbone(x)))


# =============================================================================
# Model 4: WaterNet - Water-Specific Segmentation Network
# Reference: Inspired by water body extraction methods in remote sensing
# =============================================================================
class WaterIndexModule(nn.Module):
    """Learnable water index computation module"""

    def __init__(self, in_channels=3):
        super(WaterIndexModule, self).__init__()
        # Learnable spectral indices for water detection
        self.index_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, 1),  # 4 learnable water indices
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.index_conv(x)


class WaterNet(nn.Module):
    """
    WaterNet: Specialized network for water body segmentation
    Incorporates learnable spectral indices inspired by NDWI/MNDWI
    """

    def __init__(self, n_classes=1):
        super(WaterNet, self).__init__()

        # Water index branch
        self.water_index = WaterIndexModule(3)

        # Main encoder (takes RGB + water indices)
        self.enc1 = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck with water-specific attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.water_attention = ChannelAttention(512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.outc = nn.Sequential(nn.Conv2d(64, n_classes, 1), nn.Sigmoid())

    def forward(self, x):
        # Compute water indices
        water_idx = self.water_index(x)
        x_combined = torch.cat([x, water_idx], dim=1)

        # Encoder
        e1 = self.enc1(x_combined)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck with attention
        b = self.water_attention(self.bottleneck(self.pool3(e3)))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.outc(d1)


# =============================================================================
# Model 5: MSWNet - Multi-Scale Water Network
# =============================================================================
class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction block"""

    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, 1), nn.BatchNorm2d(out_channels // 4),
                                     nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, 3, padding=1),
                                     nn.BatchNorm2d(out_channels // 4), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, 5, padding=2),
                                     nn.BatchNorm2d(out_channels // 4), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_channels, out_channels // 4, 1),
                                     nn.BatchNorm2d(out_channels // 4), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class MSWNet(nn.Module):
    """
    Multi-Scale Water Network
    Designed for robust water segmentation across different scales
    """

    def __init__(self, n_classes=1):
        super(MSWNet, self).__init__()

        # Multi-scale encoder
        self.enc1 = MultiScaleBlock(3, 64)
        self.enc2 = MultiScaleBlock(64, 128)
        self.enc3 = MultiScaleBlock(128, 256)
        self.enc4 = MultiScaleBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True)
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.outc = nn.Sequential(nn.Conv2d(64, n_classes, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bridge(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.outc(d1)


# =============================================================================
# Model 6: HRNet-Water - High-Resolution Water Segmentation
# =============================================================================
class HRNetWater(nn.Module):
    """
    Simplified HRNet for water segmentation
    Maintains high-resolution representations throughout
    """

    def __init__(self, n_classes=1):
        super(HRNetWater, self).__init__()

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        # High-resolution branch
        self.hr_branch = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )

        # Medium-resolution branch
        self.mr_branch = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True)
        )

        # Low-resolution branch
        self.lr_branch = nn.Sequential(
            nn.Conv2d(96, 192, 3, stride=2, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True)
        )

        # Fusion
        self.mr_to_hr = nn.Sequential(
            nn.Conv2d(96, 48, 1), nn.BatchNorm2d(48),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.lr_to_hr = nn.Sequential(
            nn.Conv2d(192, 48, 1), nn.BatchNorm2d(48),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

        # Final head
        self.head = nn.Sequential(
            nn.Conv2d(144, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, n_classes, 1), nn.Sigmoid()
        )

    def forward(self, x):
        stem = self.stem(x)

        hr = self.hr_branch(stem)
        mr = self.mr_branch(stem)
        lr = self.lr_branch(mr)

        # Multi-scale fusion
        mr_up = self.mr_to_hr(mr)
        lr_up = self.lr_to_hr(lr)

        fused = torch.cat([hr, mr_up, lr_up], dim=1)
        return self.head(fused)


# =============================================================================
# Model 7: SegFormer-Lite - Transformer-based Segmentation
# =============================================================================
class MixFFN(nn.Module):
    """Mix Feed-Forward Network"""

    def __init__(self, in_features, hidden_features):
        super(MixFFN, self).__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.dwconv(self.fc1(x))))


class EfficientSelfAttention(nn.Module):
    """Efficient Self-Attention with reduction"""

    def __init__(self, dim, num_heads=8, reduction_ratio=4):
        super(EfficientSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        self.reduction = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)

        x_reduced = self.reduction(x)
        _, _, H_r, W_r = x_reduced.shape
        kv = self.kv(x_reduced).reshape(B, 2, self.num_heads, C // self.num_heads, H_r * W_r)
        k, v = kv[:, 0].permute(0, 1, 3, 2), kv[:, 1].permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.proj(out)


class SegFormerLite(nn.Module):
    """
    Lightweight SegFormer for water segmentation
    Based on transformer architecture with efficient attention
    """

    def __init__(self, n_classes=1):
        super(SegFormerLite, self).__init__()

        # Patch embedding stages
        self.patch_embed1 = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3), nn.BatchNorm2d(32), nn.GELU()
        )
        self.patch_embed2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU()
        )
        self.patch_embed3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU()
        )
        self.patch_embed4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU()
        )

        # Transformer blocks (simplified)
        self.attn1 = EfficientSelfAttention(32, num_heads=1, reduction_ratio=8)
        self.ffn1 = MixFFN(32, 128)

        self.attn2 = EfficientSelfAttention(64, num_heads=2, reduction_ratio=4)
        self.ffn2 = MixFFN(64, 256)

        self.attn3 = EfficientSelfAttention(128, num_heads=4, reduction_ratio=2)
        self.ffn3 = MixFFN(128, 512)

        # MLP decoder
        self.linear_c4 = nn.Conv2d(256, 256, 1)
        self.linear_c3 = nn.Conv2d(128, 256, 1)
        self.linear_c2 = nn.Conv2d(64, 256, 1)
        self.linear_c1 = nn.Conv2d(32, 256, 1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, 1), nn.Sigmoid()
        )

    def forward(self, x):
        B, _, H, W = x.shape

        # Encoder
        c1 = self.patch_embed1(x)
        c1 = c1 + self.attn1(c1)
        c1 = c1 + self.ffn1(c1)

        c2 = self.patch_embed2(c1)
        c2 = c2 + self.attn2(c2)
        c2 = c2 + self.ffn2(c2)

        c3 = self.patch_embed3(c2)
        c3 = c3 + self.attn3(c3)
        c3 = c3 + self.ffn3(c3)

        c4 = self.patch_embed4(c3)

        # MLP Decoder
        target_size = c1.shape[-2:]

        _c4 = F.interpolate(self.linear_c4(c4), size=target_size, mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=target_size, mode='bilinear', align_corners=False)
        _c2 = F.interpolate(self.linear_c2(c2), size=target_size, mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1)

        fused = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        out = F.interpolate(self.head(fused), size=(H, W), mode='bilinear', align_corners=False)

        return out


# =============================================================================
# Evaluation and Visualization
# =============================================================================
class ModelEvaluator:
    """Comprehensive model evaluator with visualization"""

    def __init__(self, device):
        self.device = device

    def calculate_metrics(self, pred, target, threshold=0.5):
        """Calculate evaluation metrics"""
        pred_binary = (pred > threshold).cpu().numpy().flatten()
        target_binary = target.cpu().numpy().flatten()

        accuracy = accuracy_score(target_binary, pred_binary)

        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        iou = intersection / (union + 1e-8)

        tp = intersection
        fp = np.sum(pred_binary) - tp
        fn = np.sum(target_binary) - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'accuracy': accuracy, 'iou': iou, 'precision': precision,
            'recall': recall, 'f1_score': f1
        }

    def train_model(self, model, train_loader, val_loader, epochs=20, lr=1e-4):
        """Train model with history tracking"""
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': [], 'val_accuracy': []}
        best_iou = 0

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for images, masks, _ in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                if outputs.shape != masks.shape:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss, val_metrics = 0, []
            with torch.no_grad():
                for images, masks, _ in val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = model(images)
                    if outputs.shape != masks.shape:
                        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    val_loss += criterion(outputs, masks).item()
                    for i in range(outputs.shape[0]):
                        val_metrics.append(self.calculate_metrics(outputs[i, 0], masks[i, 0]))

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_iou = np.mean([m['iou'] for m in val_metrics])
            avg_val_f1 = np.mean([m['f1_score'] for m in val_metrics])
            avg_val_accuracy = np.mean([m['accuracy'] for m in val_metrics])

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_iou'].append(avg_val_iou)
            history['val_f1'].append(avg_val_f1)
            history['val_accuracy'].append(avg_val_accuracy)

            scheduler.step(avg_train_loss)
            if avg_val_iou > best_iou:
                best_iou = avg_val_iou

            if epoch % 5 == 0:
                print(
                    f'Epoch {epoch:2d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}')

        return {'best_iou': best_iou, 'history': history}

    def evaluate_model(self, model, test_loader):
        """Evaluate model performance"""
        model.eval()
        all_metrics, inference_times = [], []

        with torch.no_grad():
            for images, masks, _ in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                start_time = time.time()
                outputs = model(images)
                if outputs.shape != masks.shape:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                inference_times.append((time.time() - start_time) / images.shape[0])
                for i in range(outputs.shape[0]):
                    all_metrics.append(self.calculate_metrics(outputs[i, 0], masks[i, 0]))

        results = {}
        for key in all_metrics[0].keys():
            results[f'mean_{key}'] = np.mean([m[key] for m in all_metrics])
            results[f'std_{key}'] = np.std([m[key] for m in all_metrics])
        results['avg_inference_time'] = np.mean(inference_times)
        return results


def generate_error_maps(models, val_loader, device, save_dir='./error_maps'):
    """
    Generate error maps and ground-truth overlays for qualitative analysis
    Addresses Reviewer 5843's concern about Fig.3 lacking error visualization
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get sample images
    sample_images, sample_masks, sample_paths = [], [], []
    for images, masks, paths in val_loader:
        sample_images.append(images)
        sample_masks.append(masks)
        sample_paths.extend(paths)
        if len(sample_paths) >= 6:
            break

    sample_images = torch.cat(sample_images, dim=0)[:6].to(device)
    sample_masks = torch.cat(sample_masks, dim=0)[:6].to(device)

    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_vis = (sample_images * std + mean).clamp(0, 1)

    n_samples = sample_images.shape[0]
    n_models = len(models)

    # Create comprehensive visualization
    fig, axes = plt.subplots(n_samples, n_models + 3, figsize=(4 * (n_models + 3), 4 * n_samples))
    fig.suptitle('Qualitative Comparison with Error Maps and Ground-Truth Overlays', fontsize=16, fontweight='bold')

    column_titles = ['Input Image', 'Ground Truth'] + list(models.keys()) + ['Error Map (Ours)']

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')

    for i in range(n_samples):
        # Input image
        img_np = images_vis[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_np)
        axes[i, 0].axis('off')

        # Ground truth
        gt_np = sample_masks[i, 0].cpu().numpy()
        axes[i, 1].imshow(gt_np, cmap='Blues', vmin=0, vmax=1)
        axes[i, 1].axis('off')

        # Model predictions
        for j, (model_name, model) in enumerate(models.items()):
            model.eval()
            with torch.no_grad():
                pred = model(sample_images[i:i + 1])
                if pred.shape[-2:] != sample_masks.shape[-2:]:
                    pred = F.interpolate(pred, size=sample_masks.shape[-2:], mode='bilinear', align_corners=False)
                pred_np = pred[0, 0].cpu().numpy()

            # Create overlay: Green=TP, Red=FP, Blue=FN
            pred_binary = (pred_np > 0.5).astype(np.float32)
            gt_binary = gt_np.astype(np.float32)

            overlay = np.zeros((*pred_np.shape, 3))
            tp = (pred_binary == 1) & (gt_binary == 1)  # True Positive - Green
            fp = (pred_binary == 1) & (gt_binary == 0)  # False Positive - Red
            fn = (pred_binary == 0) & (gt_binary == 1)  # False Negative - Blue
            tn = (pred_binary == 0) & (gt_binary == 0)  # True Negative - White/Gray

            overlay[tp] = [0.2, 0.8, 0.2]  # Green
            overlay[fp] = [0.9, 0.2, 0.2]  # Red
            overlay[fn] = [0.2, 0.2, 0.9]  # Blue
            overlay[tn] = [0.9, 0.9, 0.9]  # Light gray

            # Blend with original image
            blended = 0.4 * img_np + 0.6 * overlay
            axes[i, j + 2].imshow(blended)
            axes[i, j + 2].axis('off')

            # Calculate IoU for this sample
            intersection = np.sum(tp)
            union = np.sum(tp) + np.sum(fp) + np.sum(fn)
            sample_iou = intersection / (union + 1e-8)
            axes[i, j + 2].text(5, 20, f'IoU: {sample_iou:.3f}', fontsize=10,
                                color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        # Error map for our method (last column)
        model = models['Robust U-Net (Ours)']
        with torch.no_grad():
            pred = model(sample_images[i:i + 1])
            if pred.shape[-2:] != sample_masks.shape[-2:]:
                pred = F.interpolate(pred, size=sample_masks.shape[-2:], mode='bilinear', align_corners=False)
            pred_np = pred[0, 0].cpu().numpy()

        error_map = np.abs(pred_np - gt_np)
        axes[i, -1].imshow(error_map, cmap='hot', vmin=0, vmax=1)
        axes[i, -1].axis('off')
        mae = np.mean(error_map)
        axes[i, -1].text(5, 20, f'MAE: {mae:.4f}', fontsize=10,
                         color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=[0.2, 0.8, 0.2], label='True Positive (Water)'),
        mpatches.Patch(facecolor=[0.9, 0.2, 0.2], label='False Positive'),
        mpatches.Patch(facecolor=[0.2, 0.2, 0.9], label='False Negative'),
        mpatches.Patch(facecolor=[0.9, 0.9, 0.9], label='True Negative (Land)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_path = os.path.join(save_dir, 'error_maps_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Error maps saved to {save_path}")
    return save_path


def plot_extended_comparison(results, save_path='./extended_comparison.png'):
    """Plot comprehensive comparison results"""
    methods = list(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Extended Baseline Comparison Results', fontsize=16, fontweight='bold')

    metrics = [
        ('mean_iou', 'IoU', 0, 0),
        ('mean_f1_score', 'F1-Score', 0, 1),
        ('mean_accuracy', 'Accuracy', 0, 2),
        ('mean_precision', 'Precision', 1, 0),
        ('mean_recall', 'Recall', 1, 1),
        ('avg_inference_time', 'Inference Time (ms)', 1, 2)
    ]

    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    for metric, title, row, col in metrics:
        ax = axes[row, col]
        if metric == 'avg_inference_time':
            values = [results[m][metric] * 1000 for m in methods]
        else:
            values = [results[m][metric] for m in methods]

        bars = ax.bar(range(len(methods)), values, color=colors)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(title)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)

        # Highlight best/our method
        if metric != 'avg_inference_time':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                    f'{value:.3f}' if metric != 'avg_inference_time' else f'{value:.1f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def prepare_dataset(images_dir, labels_dir, batch_size=2):
    """Prepare dataset with updated DataLoader"""
    image_files, label_files = [], []

    for img_file in sorted(os.listdir(images_dir)):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.json")
            if os.path.exists(label_path):
                image_files.append(img_path)
                label_files.append(label_path)

    print(f"Found {len(image_files)} valid image-label pairs")
    if len(image_files) == 0:
        return None

    split_idx = int(0.8 * len(image_files))
    train_imgs, val_imgs = image_files[:split_idx], image_files[split_idx:]
    train_labels, val_labels = label_files[:split_idx], label_files[split_idx:]

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CoastalDataset(train_imgs, train_labels, transform=transform)
    val_dataset = CoastalDataset(val_imgs, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def main():
    """Main comparison function with extended baselines"""
    print("=" * 80)
    print("Extended Baseline Comparison for Coastal Water Segmentation")
    print("Including Water-Specific Methods and Comprehensive Visualization")
    print("=" * 80)

    # Dataset paths
    images_dir = "./labelme_images/converted"
    labels_dir = "./labelme_images/annotations/"

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("Dataset directories not found. Please check paths.")
        return

    data_loaders = prepare_dataset(images_dir, labels_dir, batch_size=2)
    if data_loaders is None:
        return

    train_loader, val_loader = data_loaders

    # Initialize all models (including water-specific methods)
    models = {
        'Robust U-Net (Ours)': RobustUNet(n_channels=3, n_classes=1).to(device),
        'DeepLabV3+': DeepLabV3Plus(n_classes=1).to(device),
        'YOLO-SEG': YOLOSeg(n_classes=1).to(device),
        'WaterNet': WaterNet(n_classes=1).to(device),
        'MSWNet': MSWNet(n_classes=1).to(device),
        'HRNet-Water': HRNetWater(n_classes=1).to(device),
        'SegFormer-Lite': SegFormerLite(n_classes=1).to(device),
    }

    # Print model parameters
    print("\nModel Parameters:")
    print("-" * 50)
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {param_count:,} parameters ({param_count / 1e6:.2f}M)")

    evaluator = ModelEvaluator(device)
    results = {}
    training_histories = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training {name}...")
        print(f"{'=' * 60}")

        training_results = evaluator.train_model(model, train_loader, val_loader, epochs=20, lr=1e-4)
        training_histories[name] = training_results['history']

        eval_results = evaluator.evaluate_model(model, val_loader)
        results[name] = eval_results

        print(f"\nFinal Results for {name}:")
        print(f"  IoU: {eval_results['mean_iou']:.4f} ± {eval_results['std_iou']:.4f}")
        print(f"  F1:  {eval_results['mean_f1_score']:.4f} ± {eval_results['std_f1_score']:.4f}")
        print(f"  Acc: {eval_results['mean_accuracy']:.4f} ± {eval_results['std_accuracy']:.4f}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    # Error maps (addresses Reviewer 5843's concern)
    generate_error_maps(models, val_loader, device)

    # Extended comparison plot
    plot_extended_comparison(results)

    # Final summary table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Method':<20} {'IoU':<12} {'F1-Score':<12} {'Accuracy':<12} {'Params':<10} {'Time(ms)':<10}")
    print("-" * 80)

    for method_name, result in results.items():
        param_count = sum(p.numel() for p in models[method_name].parameters())
        print(f"{method_name:<20} "
              f"{result['mean_iou']:.4f}±{result['std_iou']:.3f} "
              f"{result['mean_f1_score']:.4f}±{result['std_f1_score']:.3f} "
              f"{result['mean_accuracy']:.4f}±{result['std_accuracy']:.3f} "
              f"{param_count / 1e6:.1f}M     "
              f"{result['avg_inference_time'] * 1000:.1f}")

    # Winner analysis
    print("\n" + "=" * 80)
    print("WINNER ANALYSIS")
    print("=" * 80)
    best_iou = max(results.items(), key=lambda x: x[1]['mean_iou'])
    best_f1 = max(results.items(), key=lambda x: x[1]['mean_f1_score'])
    best_acc = max(results.items(), key=lambda x: x[1]['mean_accuracy'])

    print(f"  Best IoU:      {best_iou[0]} ({best_iou[1]['mean_iou']:.4f})")
    print(f"  Best F1-Score: {best_f1[0]} ({best_f1[1]['mean_f1_score']:.4f})")
    print(f"  Best Accuracy: {best_acc[0]} ({best_acc[1]['mean_accuracy']:.4f})")

    print("\nComparison complete! Results saved to:")
    print("  - ./error_maps/error_maps_comparison.png")
    print("  - ./extended_comparison.png")


if __name__ == "__main__":
    main()
