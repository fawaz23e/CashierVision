"""Train the final ResNet18 layer and save the best validation checkpoint."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch
import torchvision
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audit_dataset_manifest import audit_manifest
from cashiervision.dataset import ProduceDataset, IMAGENET_MEAN, IMAGENET_STD
from cashiervision.model import build_model
from cashiervision.training import (
    capture_rng_state, freeze_backbone, restore_rng_state, run_epoch, save_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/processed/manifest.csv")
    parser.add_argument("--classes", type=Path, default=PROJECT_ROOT / "configs/classes_mvp.txt")
    parser.add_argument("--output-dir", type=Path, help="A new run directory; resuming uses the checkpoint directory.")
    parser.add_argument("--resume", type=Path, help="Continue from last_checkpoint.pt after its completed epoch.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of additional epochs to run (default: 5).")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--smoke-test", action="store_true", help="Run one epoch with two batches per split.")
    parser.add_argument("--no-pretrained", action="store_true", help="Offline smoke test with random weights.")
    args = parser.parse_args()
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or "optimizer_state_dict" not in checkpoint:
            parser.error("Resume requires last_checkpoint.pt from the updated training script, not best_model.pt.")
        if args.resume.name != "last_checkpoint.pt":
            parser.error("Resume from the run's last_checkpoint.pt")
        if args.no_pretrained:
            parser.error("--resume restores saved weights; omit --no-pretrained")
        if args.smoke_test and not checkpoint["config"]["smoke_test"]:
            parser.error("Cannot resume a full training run in smoke-test mode")
        args.smoke_test = checkpoint["config"]["smoke_test"]
    for name, default in (("batch_size", 32), ("learning_rate", 0.001), ("seed", 42)):
        value = getattr(args, name)
        saved = checkpoint["config"][name] if checkpoint else default
        if checkpoint and value is not None and value != saved:
            parser.error(f"Resume keeps the saved {name}: {saved}; omit its override")
        setattr(args, name, saved if value is None else value)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be at least 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    if args.no_pretrained and not args.smoke_test:
        parser.error("--no-pretrained is only available with --smoke-test")

    output_dir = args.output_dir or (args.resume.parent if args.resume else None) or PROJECT_ROOT / "models" / (
        "resnet18_smoke" if args.smoke_test else "resnet18_baseline"
    )
    if args.resume and output_dir.resolve() != args.resume.parent.resolve():
        parser.error("Resume writes to the checkpoint's directory; omit --output-dir")
    if output_dir.exists() and not args.resume:
        parser.error(f"Output already exists: {output_dir}. Choose a new --output-dir.")
    errors = audit_manifest(args.manifest, args.classes)
    if errors:
        parser.error("Manifest audit failed:\n" + "\n".join(errors))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(device_name)
    datasets = {
        split: ProduceDataset(args.manifest, split=split)
        for split in ("train", "val")
    }
    train_dataset = datasets["train"]
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if checkpoint:
        if checkpoint["class_to_idx"] != train_dataset.class_to_idx:
            parser.error("Checkpoint class mapping differs from the manifest")
        if checkpoint["config"]["manifest_sha256"] != manifest_hash:
            parser.error("Manifest changed since this checkpoint; use the original manifest to resume")
    if {label for _, label in train_dataset.samples} != set(train_dataset.class_to_idx.values()):
        parser.error("Every manifest class needs at least one training image")
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=split == "train",
            num_workers=0, generator=torch.Generator().manual_seed(args.seed),
        )
        for split, dataset in datasets.items()
    }
    model = build_model(
        len(train_dataset.class_names), pretrained=checkpoint is None and not args.no_pretrained,
    ).to(device)
    freeze_backbone(model)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    start_epoch = checkpoint["epoch"] if checkpoint else 0
    epochs = start_epoch + (1 if args.smoke_test else args.epochs)
    max_batches = 2 if args.smoke_test else None
    config = {
        "architecture": "resnet18",
        "weights": None if args.no_pretrained else "IMAGENET1K_V1",
        "frozen_backbone": True,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "smoke_test": args.smoke_test,
        "max_batches": max_batches,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "image_size": 224,
        "normalize_mean": list(IMAGENET_MEAN),
        "normalize_std": list(IMAGENET_STD),
    }
    if checkpoint:
        config["weights"] = checkpoint["config"]["weights"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"], loaders, device)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    history = checkpoint["history"] if checkpoint else []
    best_model = checkpoint["best_model"] if checkpoint else None
    best_accuracy = best_model["validation"]["top1_accuracy"] if best_model else -1.0
    best_epoch = best_model["epoch"] if best_model else 0
    if checkpoint:
        # The latest checkpoint is authoritative if a previous export was interrupted.
        save_checkpoint(best_model, output_dir / "best_model.pt")
        print(f"Resuming after epoch {start_epoch}; next epoch is {start_epoch + 1}", flush=True)
    print(f"Device: {device}; classes: {len(train_dataset.class_names)}; training only model.fc", flush=True)
    if args.smoke_test:
        print("Smoke test only: these metrics are not a baseline result.", flush=True)

    for epoch in range(start_epoch + 1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}: training", flush=True)
        train_metrics = run_epoch(model, loaders["train"], device, optimizer, max_batches)
        print("Validating", flush=True)
        val_metrics = run_epoch(model, loaders["val"], device, max_batches=max_batches)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["top1_accuracy"] > best_accuracy:
            best_accuracy = val_metrics["top1_accuracy"]
            best_epoch = epoch
            best_model = {
                "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "class_names": train_dataset.class_names,
                "class_to_idx": train_dataset.class_to_idx,
                "epoch": epoch,
                "validation": val_metrics,
                "config": config,
            }
        save_checkpoint({
            "format_version": 1,
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": capture_rng_state(loaders, device),
            "class_to_idx": train_dataset.class_to_idx,
            "epoch": epoch,
            "config": config,
            "history": history,
            "best_model": best_model,
        }, output_dir / "last_checkpoint.pt")
        save_checkpoint(best_model, output_dir / "best_model.pt")
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump({"config": config, "best_epoch": best_epoch, "history": history}, file, indent=2, allow_nan=False)
        print(
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val top-1: {val_metrics['top1_accuracy']:.1%} | "
            f"Val top-3: {val_metrics['top3_accuracy']:.1%}", flush=True,
        )
    print(f"Saved best epoch ({best_epoch}) and metrics to {output_dir}")
    print("The test split has not been evaluated.")


if __name__ == "__main__":
    main()
