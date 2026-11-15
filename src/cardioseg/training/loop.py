"""The training loop: show the model a batch, measure how wrong it is, nudge every weight, repeat.

One STEP   = one batch: forward -> loss -> backward -> optimizer step.
One EPOCH  = one pass over every training slice.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader


def pick_device() -> torch.device:
    """Use the Mac GPU (MPS) if there is one, an NVIDIA GPU (CUDA) if there is one, else the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass over the training data. Returns the average loss."""
    model.train()  # training mode: BatchNorm learns from each batch
    total = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()  # 0. forget the previous batch's gradients
        scores = model(images)  # 1. forward: make a guess
        loss = loss_fn(scores, masks)  # 2. loss: how wrong was it? (one number)
        loss.backward()  # 3. backward: gradient for every weight
        optimizer.step()  # 4. step: nudge every weight a little downhill

        total += loss.item()
    return total / len(loader)


@torch.no_grad()  # just checking, not learning: skip gradient bookkeeping
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> float:
    """Average loss on data the model is NOT trained on. Weights are never changed here."""
    model.eval()  # evaluation mode: BatchNorm uses what it learned, not this batch
    total = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        total += loss_fn(model(images), masks).item()
    return total / len(loader)
