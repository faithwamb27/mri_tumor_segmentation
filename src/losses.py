"""
Loss functions for multi-class 3-D segmentation.

• SoftDiceLoss   – per-class soft Dice, averaged over foreground classes
• CombinedLoss   – weighted sum of SoftDiceLoss + CrossEntropyLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss averaged over foreground classes.

    Parameters
    ----------
    num_classes : total number of classes (including background at index 0)
    smooth      : Laplace smoothing constant
    skip_bg     : if True, background (class 0) is excluded from the average
    """

    def __init__(self, num_classes: int = 4,
                 smooth: float = 1e-5, skip_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.skip_bg     = skip_bg

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, D, H, W)  raw model output (before softmax)
        targets : (B, D, H, W)     integer class labels
        """
        probs  = F.softmax(logits, dim=1)            # (B, C, D, H, W)
        # One-hot encode targets → (B, C, D, H, W)
        oh     = F.one_hot(targets, self.num_classes).permute(0, 4, 1, 2, 3).float()

        start  = 1 if self.skip_bg else 0
        dice_per_class = []
        for c in range(start, self.num_classes):
            p = probs[:, c]           # (B, D, H, W)
            g = oh[:, c]              # (B, D, H, W)
            intersection = (p * g).sum()
            denom        = p.sum() + g.sum() + self.smooth
            dice_per_class.append(1.0 - (2.0 * intersection + self.smooth) / denom)

        return torch.stack(dice_per_class).mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of Dice loss and cross-entropy loss.

    L = dice_w * DiceLoss + ce_w * CrossEntropyLoss
    """

    def __init__(
        self,
        num_classes: int  = 4,
        dice_weight: float = 0.5,
        ce_weight: float   = 0.5,
        smooth: float      = 1e-5,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.dice_w  = dice_weight
        self.ce_w    = ce_weight
        self.dice    = SoftDiceLoss(num_classes, smooth)
        self.ce      = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, D, H, W)
        targets : (B, D, H, W)  int64
        """
        loss_dice = self.dice(logits, targets)
        loss_ce   = self.ce(logits, targets)
        return self.dice_w * loss_dice + self.ce_w * loss_ce


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loss(config) -> nn.Module:
    """Create CombinedLoss from a Config object."""
    return CombinedLoss(
        num_classes  = config.num_classes,
        dice_weight  = config.loss_dice_weight,
        ce_weight    = config.loss_ce_weight,
        smooth       = config.dice_smooth,
    )
