#!/usr/bin/env python3
"""
Model factory for PCG-based LVEF screening.

Supported backbones (ImageNet-pretrained via timm):
    - MobileNetV2
    - MobileNetV3-Large
    - EfficientNet-B0
    - EfficientNetV2-S
    - SwinV2-Tiny
    - SwinV2-Small
"""

from typing import Dict

import timm
import torch
import torch.nn as nn

# Map friendly names to timm model identifiers
BACKBONE_CONFIGS: Dict[str, str] = {
    "mobilenetv2": "mobilenetv2_100",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "swinv2_tiny": "swinv2_tiny_window8_256",
    "swinv2_small": "swinv2_small_window8_256",
}


def create_model(
    backbone: str = "mobilenetv2",
    pretrained: bool = True,
    num_classes: int = 1,
) -> nn.Module:
    """
    Create an ImageNet-pretrained model with a single-logit head.

    Args:
        backbone: Friendly backbone name (see BACKBONE_CONFIGS).
        pretrained: Whether to load ImageNet weights.
        num_classes: Output dimension (1 for BCEWithLogitsLoss).
    """
    if backbone not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available: {list(BACKBONE_CONFIGS.keys())}"
        )

    model_name = BACKBONE_CONFIGS[backbone]

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=3,
        num_classes=num_classes,
    )
    return model


class MILAttentionPool(nn.Module):
    """Gated attention pooling for MIL bags."""

    def __init__(self, in_dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.attn_v = nn.Linear(in_dim, hidden_dim)
        self.attn_u = nn.Linear(in_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, feats: torch.Tensor, mask: torch.Tensor = None):
        # feats: [B, N, D]
        attn_v = torch.tanh(self.attn_v(feats))
        attn_u = torch.sigmoid(self.attn_u(feats))
        scores = self.attn_w(attn_v * attn_u).squeeze(-1)  # [B, N]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * feats, dim=1)
        if self.dropout is not None:
            pooled = self.dropout(pooled)
        return pooled, attn


class MILModel(nn.Module):
    """Backbone feature extractor + attention pooling head."""

    def __init__(
        self,
        backbone: str = "mobilenetv2",
        pretrained: bool = True,
        attn_hidden: int = None,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if backbone not in BACKBONE_CONFIGS:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Available: {list(BACKBONE_CONFIGS.keys())}"
            )
        model_name = BACKBONE_CONFIGS[backbone]
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.pool = MILAttentionPool(
            in_dim=feat_dim, hidden_dim=attn_hidden, dropout=attn_dropout
        )
        self.classifier = nn.Linear(feat_dim, 1)

    def forward(self, bags: torch.Tensor, mask: torch.Tensor = None):
        # bags: [B, N, C, H, W]
        bsz, n_inst, c, h, w = bags.shape
        feats = self.backbone(bags.view(bsz * n_inst, c, h, w))
        feats = feats.view(bsz, n_inst, -1)
        pooled, attn = self.pool(feats, mask=mask)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, attn
