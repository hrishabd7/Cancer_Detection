import math
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int, feature_dim: int | None = 512, pretrained: bool = True) -> nn.Module:
    """
    Return a torchvision convnext_tiny with the final linear adjusted for num_classes.
    Default behavior: create a 512-d bottleneck head (768 -> 512 -> num_classes).
    If feature_dim is None or equals the backbone in_feats, keep a single final linear layer.
    """
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.convnext_tiny(weights=weights)
    in_feats = m.classifier[-1].in_features  # backbone penultimate dim (768)

    if feature_dim is None or feature_dim == in_feats:
        # keep default single linear layer
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
    else:
        # keep avgpool and flatten from the original classifier, then add projection head
        avgpool = m.classifier[0]   # AdaptiveAvgPool2d((1,1))
        flatten = m.classifier[1]   # Flatten()
        m.classifier = nn.Sequential(
            avgpool,
            flatten,
            nn.Linear(in_feats, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_classes)
        )
    return m

# def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
#     """
#     Return a torchvision convnext_tiny with the final linear adjusted for num_classes.
#     If pretrained is True, ImageNet weights are loaded.
#     """
#     weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
#     m = models.convnext_tiny(weights=weights)
#     in_feats = m.classifier[-1].in_features
#     m.classifier[-1] = nn.Linear(in_feats, num_classes)
#     return m

# class CustomConvNeXtTiny(nn.Module):
#     def __init__(self, num_classes, feature_dim=768):
#         super().__init__()
#         self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
#         in_feats = self.backbone.classifier[-1].in_features
#         self.backbone.classifier = nn.Identity()
#         self.classifier = nn.Sequential(
#             nn.Linear(in_feats, feature_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim, num_classes)
#         )

#     def forward(self, x):
#         x = self.backbone.features(x)
#         x = self.backbone.avgpool(x)
#         x = x.flatten(1)
#         x = self.classifier(x)
#         return x

# def build_model(num_classes: int, feature_dim: int = 768) -> nn.Module:
#     return CustomConvNeXtTiny(num_classes, feature_dim)

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = max(warmup_epochs, 0)
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        t = self.last_epoch - self.warmup_epochs
        T = max(1, self.total_epochs - self.warmup_epochs)
        return [base_lr * 0.5 * (1 + math.cos(math.pi * t / T))
                for base_lr in self.base_lrs]

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, target):
        logp = nn.functional.log_softmax(logits, dim=1)
        p = logp.exp()
        ce = nn.functional.nll_loss(logp, target, weight=self.weight, reduction='none')
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()