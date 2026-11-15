"""Loss functions: one number that says how wrong the model's scores are, compared with the real mask.

Cross-entropy: per pixel, how much probability did the model give the right class?
    Steady and well-behaved, but every pixel counts equally, and ~90% of pixels are background.
Dice: per class, how much do the predicted and real regions overlap?
    Small structures (RV, myocardium) count as much as big ones, but it is jumpy early on.
Combined = cross-entropy + Dice: the steadiness of one plus the small-structure focus of the other.
"""

import torch
import torch.nn.functional as F
from torch import nn


def soft_dice_loss(
    logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """1 - average Dice over the heart classes (1, 2, 3), using probabilities, not hard labels.

    logits: (B, C, H, W) raw scores.  mask: (B, H, W) real labels 0..C-1.
    """
    num_classes = logits.shape[1]
    probs = torch.softmax(
        logits, dim=1
    )  # scores -> probabilities that sum to 1 per pixel
    target = (
        F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float()
    )  # (B, C, H, W) of 0s and 1s

    dims = (0, 2, 3)  # add up over the batch and every pixel, separately for each class
    overlap = (probs * target).sum(dims)
    total = probs.sum(dims) + target.sum(dims)
    dice = (2 * overlap + eps) / (total + eps)  # 1 = perfect overlap, 0 = none
    return (
        1 - dice[1:].mean()
    )  # skip class 0: background is easy and would hide mistakes


class CombinedLoss(nn.Module):
    """Cross-entropy + Dice. Turn Dice off with use_dice=False (that is its own ablation row)."""

    def __init__(self, use_dice: bool = True):
        super().__init__()
        self.use_dice = use_dice

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(logits, mask)
        if self.use_dice:
            loss = loss + soft_dice_loss(logits, mask)
        return loss
