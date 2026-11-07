"""
ImageNet-pretrained ResNet-50 with dynamic multi-head management for the
applied fine-tuning regime (Scenario B: ISIC 2019 dermoscopy + Imagenette).

Architecture:
- Backbone: torchvision ResNet-50 initialized from IMAGENET1K_V2 weights.
- Penultimate representations: 2048-dimensional embedding.
- Head A: Linear(2048, 8) for Target Task A (ISIC 2019).
- Head B: Dynamic Linear(2048, 10) for Auxiliary Task B (Imagenette).
- Head Temp: Dynamic Linear(2048, 8) for Condition C3 (On-target reset control).
- BatchNorm policy: Running statistics are preserved across phases (no boundary reset).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class PretrainedResNet50DualHead(nn.Module):
    """
    ResNet-50 pretrained on ImageNet-1K (V2 weights) with dual-head boundary switching.
    """
    def __init__(self, num_classes_a=8, weights=ResNet50_Weights.IMAGENET1K_V2):
        super().__init__()
        # Load torchvision pretrained backbone
        base_model = models.resnet50(weights=weights)

        # Retain all layers except fc
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool
        self.feat_dim = 2048
        self.num_classes_a = num_classes_a

        # Primary classification head for Target Task A
        self.head_a = nn.Linear(self.feat_dim, num_classes_a)
        nn.init.kaiming_normal_(self.head_a.weight, nonlinearity='linear')
        nn.init.constant_(self.head_a.bias, 0.0)

        # Dynamic heads instantiated during Phase 2
        self.head_b = None
        self.head_temp = None

    def extract_features(self, x):
        """Extract penultimate 2048-dimensional representations."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        return features

    def forward(self, x, head='A'):
        """Forward pass with specified active head."""
        feat = self.extract_features(x)
        if head == 'A':
            return self.head_a(feat)
        elif head == 'B':
            if self.head_b is None:
                raise RuntimeError("Head B requested but not initialized.")
            return self.head_b(feat)
        elif head == 'temp':
            if self.head_temp is None:
                raise RuntimeError("Head Temp requested but not initialized.")
            return self.head_temp(feat)
        elif head == 'features':
            return feat
        else:
            raise ValueError(f"Invalid head argument: {head}")

    def get_shared_parameters(self):
        """Returns generator over shared backbone parameters (excluding all heads)."""
        for name, param in self.named_parameters():
            if not name.startswith("head_"):
                yield param

    def init_auxiliary_head(self, num_classes_b=10, device='cpu'):
        """Initializes fresh temporary head_b for Phase 2 auxiliary detour (S1)."""
        self.head_b = nn.Linear(self.feat_dim, num_classes_b).to(device)
        nn.init.kaiming_normal_(self.head_b.weight, nonlinearity='linear')
        nn.init.constant_(self.head_b.bias, 0.0)

    def init_temp_head_a(self, device='cpu'):
        """Initializes fresh temporary head_temp with |Y_A| classes for Condition C3."""
        self.head_temp = nn.Linear(self.feat_dim, self.num_classes_a).to(device)
        nn.init.kaiming_normal_(self.head_temp.weight, nonlinearity='linear')
        nn.init.constant_(self.head_temp.bias, 0.0)

    def freeze_head_a(self):
        """Freezes Head A parameters so they receive 0 gradients during Phase 2."""
        for param in self.head_a.parameters():
            param.requires_grad = False

    def unfreeze_head_a(self):
        """Unfreezes Head A parameters for Phase 3 target relearning."""
        for param in self.head_a.parameters():
            param.requires_grad = True

    def discard_auxiliary_head(self):
        """Permanently discards temporary Head B and Head Temp."""
        self.head_b = None
        self.head_temp = None
