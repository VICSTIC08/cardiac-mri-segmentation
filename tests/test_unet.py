"""Shape tests: if the U-Net's sides don't line up, these fail before any training starts."""

import torch

from cardioseg.models.unet import AttentionGate, ResidualBlock, UNet


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


def test_residual_unet_has_the_same_output_shape():
    model = UNet(base_channels=8, residual=True)
    assert model(torch.randn(2, 1, 224, 224)).shape == (2, 4, 224, 224)


def test_residual_block_passes_its_input_through_when_it_has_learned_nothing():
    """If the convolutions add nothing, the shortcut still delivers x: output = ReLU(0 + x)."""
    block = ResidualBlock(8, 8)
    for param in block.layers.parameters():
        torch.nn.init.zeros_(param)
    block.eval()
    x = torch.randn(1, 8, 16, 16)
    assert torch.allclose(block(x), torch.relu(x))


def test_attention_unet_has_the_same_output_shape():
    model = UNet(base_channels=8, attention=True)
    assert model(torch.randn(2, 1, 224, 224)).shape == (2, 4, 224, 224)


def test_attention_gate_scores_every_pixel_between_0_and_1():
    gate = AttentionGate(16)
    decoder, skip = torch.randn(2, 16, 28, 28), torch.randn(2, 16, 28, 28)
    gated = gate(decoder, skip)
    assert gated.shape == skip.shape
    assert gate.last_map.shape == (2, 1, 28, 28)  # one score per pixel
    assert gate.last_map.min() >= 0 and gate.last_map.max() <= 1


def test_all_options_together():
    model = UNet(base_channels=8, residual=True, attention=True)
    assert model(torch.randn(1, 1, 64, 64)).shape == (1, 4, 64, 64)
