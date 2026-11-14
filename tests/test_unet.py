"""Shape tests: if the U-Net's sides don't line up, these fail before any training starts."""

import torch

from cardioseg.models.unet import UNet


def test_output_has_one_score_map_per_class():
    model = UNet(in_channels=1, num_classes=4, base_channels=8)
    images = torch.randn(2, 1, 224, 224)
    scores = model(images)
    assert scores.shape == (2, 4, 224, 224)


def test_scores_can_be_compared_with_a_mask():
    """Each pixel's best-scoring class gives a predicted mask the same shape as a real one."""
    model = UNet(base_channels=8)
    prediction = model(torch.randn(2, 1, 224, 224)).argmax(dim=1)
    assert prediction.shape == (2, 224, 224)
    assert set(prediction.unique().tolist()) <= {0, 1, 2, 3}


def test_every_layer_can_learn():
    """After one backward pass, every weight has a gradient, so nothing is accidentally disconnected."""
    model = UNet(base_channels=8)
    model(torch.randn(2, 1, 64, 64)).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
