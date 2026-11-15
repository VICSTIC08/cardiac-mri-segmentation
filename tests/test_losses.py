"""Known-answer tests: a perfect guess should cost ~0, a wrong guess a lot, a coin-flip guess ln(4)."""

import math

import torch

from cardioseg.training.losses import CombinedLoss, soft_dice_loss


def make_mask():
    mask = torch.zeros(2, 32, 32, dtype=torch.long)
    mask[:, 8:24, 8:24] = 2  # myocardium square
    mask[:, 12:20, 12:20] = 3  # LV inside it
    mask[:, 2:6, 2:10] = 1  # small RV
    return mask


def confident_scores(labels, num_classes=4):
    """Scores that are very sure of the given labels: 20 for that class, 0 for the rest."""
    return (
        torch.nn.functional.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
        * 20
    )


def test_perfect_prediction_has_almost_zero_loss():
    mask = make_mask()
    assert soft_dice_loss(confident_scores(mask), mask) < 0.01
    assert CombinedLoss()(confident_scores(mask), mask) < 0.01


def test_wrong_prediction_has_high_dice_loss():
    mask = make_mask()
    everything_background = confident_scores(torch.zeros_like(mask))
    assert soft_dice_loss(everything_background, mask) > 0.99


def test_unsure_model_gets_cross_entropy_of_ln_4():
    """Equal scores = 25% for each of 4 classes, and cross-entropy of a 25% guess is ln(4)."""
    mask = make_mask()
    unsure = torch.zeros(2, 4, 32, 32)
    loss = CombinedLoss(use_dice=False)(unsure, mask)
    assert math.isclose(loss.item(), math.log(4), rel_tol=1e-5)


def test_loss_gives_every_score_a_gradient():
    mask = make_mask()
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    CombinedLoss()(logits, mask).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
