"""Training and validation for a ResNet with a frozen feature extractor."""

from itertools import islice
from pathlib import Path
import os
import random
import tempfile

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.models import ResNet


def save_checkpoint(checkpoint: dict, path: Path) -> None:
    """Replace a checkpoint only after its new contents are fully written."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary_path = Path(file.name)
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def capture_rng_state(loaders: dict[str, DataLoader], device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "loaders": {name: loader.generator.get_state() for name, loader in loaders.items()},
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict, loaders: dict[str, DataLoader], device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    for name, loader in loaders.items():
        loader.generator.set_state(state["loaders"][name])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def freeze_backbone(model: ResNet) -> None:
    """Allow only the final classification layer to learn."""
    model.requires_grad_(False)
    model.fc.requires_grad_(True)


def run_epoch(
    model: ResNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Train when given an optimizer; otherwise only measure performance."""
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1")

    # Frozen BatchNorm layers must also keep their learned running statistics.
    model.eval()
    model.fc.train(optimizer is not None)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = correct_top1 = correct_top3 = 0
    batches = loader if max_batches is None else islice(loader, max_batches)

    for batch_number, (images, labels) in enumerate(batches, start=1):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            loss = loss_fn(scores, labels)
            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite loss encountered")
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        guesses = scores.detach().topk(min(3, scores.size(1)), dim=1).indices
        matches = guesses.eq(labels.unsqueeze(1))
        correct_top1 += matches[:, 0].sum().item()
        correct_top3 += matches.any(dim=1).sum().item()
        if batch_number % 50 == 0:
            print(f"  Processed {batch_number} batches ({total_samples} images)", flush=True)

    if total_samples == 0:
        raise ValueError("Cannot measure an empty data loader")
    return {
        "loss": total_loss / total_samples,
        "top1_accuracy": correct_top1 / total_samples,
        "top3_accuracy": correct_top3 / total_samples,
        "samples": total_samples,
    }
