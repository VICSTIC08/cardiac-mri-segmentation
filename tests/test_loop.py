"""The most important sanity check in deep learning: can the model memorise ONE tiny batch?

If the loss does not drop on 4 images it sees over and over, something is broken
(wrong shapes, wrong labels, a disconnected layer) and a long training run would be wasted.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from cardioseg.models.unet import UNet
from cardioseg.training.loop import evaluate, train_one_epoch
from cardioseg.training.losses import CombinedLoss


def tiny_dataset():
    torch.manual_seed(0)
    masks = torch.zeros(4, 32, 32, dtype=torch.long)
    masks[:, 8:24, 8:24] = 2
    masks[:, 12:20, 12:20] = 3
    images = masks.unsqueeze(1).float() + 0.1 * torch.randn(4, 1, 32, 32)
    return TensorDataset(images, masks)


def test_model_can_overfit_one_tiny_batch():
    torch.manual_seed(0)
    loader = DataLoader(tiny_dataset(), batch_size=4)
    model = UNet(base_channels=8)
    loss_fn = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    device = torch.device("cpu")

    first = train_one_epoch(model, loader, loss_fn, optimizer, device)
    for _ in range(30):
        last = train_one_epoch(model, loader, loss_fn, optimizer, device)
    assert last < first / 3


def test_evaluate_does_not_change_the_weights():
    loader = DataLoader(tiny_dataset(), batch_size=4)
    model = UNet(base_channels=8)
    before = [p.clone() for p in model.parameters()]
    evaluate(model, loader, CombinedLoss(), torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
