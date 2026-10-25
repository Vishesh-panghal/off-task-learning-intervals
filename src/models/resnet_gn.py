"""
ResNet-18 with Group Normalization (G=32) for CIFAR-scale 32x32 inputs.
Features dynamic dual-head management for SOTLI (Task A and Task B classification heads).
Zero BatchNorm layers exist in this model, completely eliminating running-statistics drift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockGN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_groups=32):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(num_groups, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(num_groups, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18GroupNorm(nn.Module):
    """
    ResNet-18 with GroupNorm (G=32) tailored for 32x32 resolution images.
    Initial 3x3 conv with stride 1, avoiding the standard 7x7 stride 2 + maxpool
    which degrades spatial features prematurely on 32x32 inputs.
    """
    def __init__(self, num_classes_a=100, num_groups=32):
        super().__init__()
        self.inplanes = 64
        self.num_groups = num_groups
        self.feat_dim = 512

        # 3x3 conv with stride 1 for 32x32 inputs
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, 64)
        self.relu = nn.ReLU(inplace=True)

        # 4 Residual Stages: [2, 2, 2, 2] BasicBlocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dedicated Classification Heads
        self.head_a = nn.Linear(512, num_classes_a)
        self.head_b = None  # Instantiated dynamically during Phase 2

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlockGN.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlockGN.expansion, stride),
                nn.GroupNorm(self.num_groups, planes * BasicBlockGN.expansion),
            )

        layers = []
        layers.append(BasicBlockGN(self.inplanes, planes, stride, downsample, self.num_groups))
        self.inplanes = planes * BasicBlockGN.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlockGN(self.inplanes, planes, num_groups=self.num_groups))

        return nn.Sequential(*layers)

    def extract_features(self, x):
        """Extract penultimate representations (512-dim embedding)"""
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        return features

    def forward(self, x, head='A'):
        """Forward pass with specified active head"""
        feat = self.extract_features(x)
        if head == 'A':
            return self.head_a(feat)
        elif head == 'B':
            if self.head_b is None:
                raise RuntimeError("Head B requested but not initialized.")
            return self.head_b(feat)
        elif head == 'features':
            return feat
        else:
            raise ValueError(f"Invalid head argument: {head}")

    def get_shared_parameters(self):
        """Returns generator over shared backbone parameters w (excluding all heads)"""
        for name, param in self.named_parameters():
            if not name.startswith("head_"):
                yield param

    def init_auxiliary_head(self, num_classes_b, device='cpu'):
        """Initializes fresh temporary head_B for Phase 2"""
        self.head_b = nn.Linear(self.feat_dim, num_classes_b).to(device)
        nn.init.kaiming_normal_(self.head_b.weight, nonlinearity='linear')
        nn.init.constant_(self.head_b.bias, 0)

    def freeze_head_a(self):
        """Freezes Head A parameters so they receive 0 gradients during Phase 2"""
        for param in self.head_a.parameters():
            param.requires_grad = False

    def unfreeze_head_a(self):
        """Unfreezes Head A parameters for Phase 3 target relearning"""
        for param in self.head_a.parameters():
            param.requires_grad = True

    def discard_auxiliary_head(self):
        """Permanently discards temporary Head B"""
        self.head_b = None
