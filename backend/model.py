"""Serving-side model definitions.

Two architectures exist in this project's history and they are NOT interchangeable:

* ``plain``     - timm ``efficientnet_b3`` with a single linear classifier.
                  This is ``models/latest.pt``, the checkpoint currently served,
                  and the one whose measured test-set performance is documented.
* ``multihead`` - EfficientNet-B3 backbone plus the two-layer head
                  (1536 -> 512 -> 256 -> 7) from ``skin_cancer/src/model.py``.
                  This is ``skin_cancer/checkpoints/best.pt``, an earlier and
                  substantially weaker run (measured macro-F1 0.45 vs 0.77).

The architecture is selected explicitly by ``MODEL_ARCH`` rather than inferred
from the checkpoint, and ``load_state_dict`` is strict, so a mismatch fails loudly
instead of silently serving a different network than the metrics describe.
"""

import timm
import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Two-layer classifier head with independent dropout gates."""

    def __init__(self, in_features: int, num_classes: int,
                 hidden_dim: int = 512, drop_rate: float = 0.5):
        super().__init__()
        mid_dim = hidden_dim // 2  # 512 -> 256
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SkinCancerModel(nn.Module):
    """EfficientNet-B3 backbone with the multi-layer classifier head."""

    def __init__(self, backbone_name: str = "efficientnet_b3", num_classes: int = 7,
                 pretrained: bool = False, drop_rate: float = 0.5,
                 head_hidden_dim: int = 512, drop_path_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # head supplied separately
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.classifier = ClassifierHead(
            in_features=self.backbone.num_features,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            drop_rate=drop_rate,
        )

    @property
    def conv_head(self) -> nn.Module:
        """Final conv layer, the Grad-CAM target."""
        return self.backbone.conv_head

    def pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone features after pooling — the classifier head's input."""
        features = self.backbone.forward_features(x)
        if features.dim() == 4:  # (B, C, H, W)
            features = features.mean(dim=[2, 3])
        elif features.dim() == 3:  # (B, tokens, C)
            features = features.mean(dim=1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pooled_features(x))


def build_model(arch: str = "plain", num_classes: int = 7) -> nn.Module:
    """Constructs the architecture named by MODEL_ARCH.

    The plain model is returned unwrapped so its state-dict keys match the
    checkpoint exactly; use the helpers below rather than reaching into it.
    """
    if arch == "plain":
        return timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=num_classes)
    if arch == "multihead":
        return SkinCancerModel(num_classes=num_classes, pretrained=False)
    raise ValueError(
        "Unknown MODEL_ARCH {!r}. Expected 'plain' or 'multihead'.".format(arch))


def get_conv_head(model: nn.Module) -> nn.Module:
    """Final conv layer of either architecture — the Grad-CAM target."""
    return model.conv_head


def get_pooled_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Penultimate pooled features of either architecture."""
    if hasattr(model, "pooled_features"):          # multihead
        return model.pooled_features(x)
    return model.forward_head(model.forward_features(x), pre_logits=True)  # plain
