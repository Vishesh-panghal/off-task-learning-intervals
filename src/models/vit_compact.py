"""
Compact Vision Transformer (ViT) for CIFAR-scale 32x32 inputs.
Features dynamic dual-head management matching the SOTLI ResNet-18 interface.
Uses LayerNorm, completely eliminating running-statistics drift.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactViT(nn.Module):
    """
    Compact Vision Transformer tailored for 32x32 image inputs.
    Patch size 4x4 gives 8x8 = 64 patches.
    Embed dim 192, 6 transformer layers, 4 attention heads.
    Uses LayerNorm (Norm-First design) for stable training.
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes_a=100,
                 embed_dim=192, depth=6, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.feat_dim = embed_dim

        # Patch embedding via non-overlapping 4x4 convolution
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable class token and 1D learnable position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Pre-LN Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Dedicated Classification Heads (matching ResNet18GroupNorm)
        self.head_a = nn.Linear(embed_dim, num_classes_a)
        self.head_b = None  # Instantiated dynamically during Phase 2

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        """Extract penultimate representation vector (192-dim CLS token embedding)"""
        B = x.shape[0]
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        features = self.norm(x[:, 0])  # CLS token representation
        return features

    def forward(self, x, head='A'):
        """Forward pass matching ResNet18GroupNorm interface"""
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
        """Returns generator over shared backbone parameters (excluding heads)"""
        for name, param in self.named_parameters():
            if not name.startswith("head_"):
                yield param

    def init_auxiliary_head(self, num_classes_b, device='cpu'):
        """Initializes fresh temporary head_B for Phase 2"""
        self.head_b = nn.Linear(self.feat_dim, num_classes_b).to(device)
        nn.init.trunc_normal_(self.head_b.weight, std=0.02)
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
