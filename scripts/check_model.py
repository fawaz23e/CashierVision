"""Pass a batch of real images through the CNN without training it."""

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cashiervision.dataset import ProduceDataset
from cashiervision.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "manifest.csv",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Use random weights for an offline connection check.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    dataset = ProduceDataset(args.manifest, split="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = build_model(len(dataset.class_names), pretrained=not args.no_pretrained)
    model.eval()

    images, labels = next(iter(loader))
    with torch.inference_mode():
        scores = model(images)

    expected_shape = (len(labels), len(dataset.class_names))
    if tuple(scores.shape) != expected_shape or not torch.isfinite(scores).all():
        raise RuntimeError("Model returned an invalid score batch")

    print(f"Classes: {len(dataset.class_names)}")
    print(f"Image batch: {tuple(images.shape)}")
    print(f"Label batch: {tuple(labels.shape)}")
    print(f"Model scores: {tuple(scores.shape)}")
    print("Connection check passed. The produce classifier has not been trained.")
    if args.no_pretrained:
        print("Offline check: all model weights were randomly initialized.")


if __name__ == "__main__":
    main()
