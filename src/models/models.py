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
