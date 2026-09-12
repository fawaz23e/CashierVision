"""Evaluate an already-trained checkpoint on a manifest. Read-only: never trains.

Use this to ask "how does the Fruits-360 model cope with real store photos?"
The model keeps its original 18 outputs; nothing is retrained or fine-tuned.

Labels are matched BY NAME, not by index. The baseline model's class order is
  banana, cantaloupe, cauliflower, eggplant, ...
and the grocery manifest's order is
  banana, cantaloupe, eggplant, ginger_root, ...
so they agree at index 0 and 1 then silently diverge. Comparing raw integers
would score eggplant against cauliflower and look like a real result.

Two accuracy numbers are reported:

  open      argmax over all 18 outputs. The honest deployment number -- the
            model may answer "cauliflower" even though this manifest has none.
  restricted argmax after hiding outputs absent from the manifest. Answers the
            narrower question "can it separate the 14 classes it is being shown?"
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cashiervision.dataset import ProduceDataset
from src.cashiervision.model import build_model


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    return torch.device(name)


@torch.no_grad()
def evaluate(model, loader, device, dataset_to_model, keep_columns, model_classes, k=3):
    """Return per-image predictions across the full top-k x label-space grid.

    Two independent axes:
      top-1 vs top-k   -- how many guesses we allow. A cashier assist shows a
                          shortlist, so top-k is the product-facing number;
                          top-1 is the strict one.
      open vs restricted -- which label space the model may answer in. open uses
                          all of the model's classes; restricted hides classes
                          absent from this manifest before ranking.
    """
    model.eval()
    n_kept = int(keep_columns.sum())
    open_k = min(k, len(model_classes))
    restricted_k = min(k, n_kept)

    records = []
    for images, dataset_labels in loader:
        images = images.to(device)
        logits = model(images)

        # Translate this batch's manifest indices into the model's index space.
        true_model_idx = dataset_to_model[dataset_labels]

        open_top = logits.topk(open_k, dim=1).indices.cpu()

        masked = logits.clone()
        masked[:, ~keep_columns] = float("-inf")
        restricted_top = masked.topk(restricted_k, dim=1).indices.cpu()

        for i in range(len(dataset_labels)):
            truth = int(true_model_idx[i])
            open_ranked = [int(j) for j in open_top[i]]
            restricted_ranked = [int(j) for j in restricted_top[i]]
            records.append(
                {
                    "true": model_classes[truth],
                    "open_top1": model_classes[open_ranked[0]],
                    "open_topk": [model_classes[j] for j in open_ranked],
                    "restricted_top1": model_classes[restricted_ranked[0]],
                    "restricted_topk": [model_classes[j] for j in restricted_ranked],
                    "correct_open_top1": open_ranked[0] == truth,
                    "correct_open_topk": truth in set(open_ranked),
                    "correct_restricted_top1": restricted_ranked[0] == truth,
                    "correct_restricted_topk": truth in set(restricted_ranked),
                }
            )
    return records


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_classes: list[str] = checkpoint["class_names"]
    model_class_to_idx: dict[str, int] = checkpoint["class_to_idx"]

    dataset = ProduceDataset(args.manifest, split=args.split)
    if len(dataset) == 0:
        raise SystemExit(f"No images in split '{args.split}' of {args.manifest}")

    unknown = sorted(set(dataset.class_names) - set(model_class_to_idx))
    if unknown:
        raise SystemExit(
            "These manifest labels are not in the checkpoint, so they can never be "
            f"predicted correctly: {unknown}"
        )

    # Index translation, built by name. Position i holds the model's index for
    # the manifest's class i.
    dataset_to_model = torch.tensor(
        [model_class_to_idx[name] for name in dataset.class_names], dtype=torch.long
    )
    keep_columns = torch.zeros(len(model_classes), dtype=torch.bool)
    keep_columns[dataset_to_model] = True

    model = build_model(num_classes=len(model_classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    records = evaluate(
        model, loader, device, dataset_to_model, keep_columns, model_classes, k=args.topk
    )

    report(records, args, checkpoint, dataset, model_classes, keep_columns, device)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "manifest": str(args.manifest),
                    "split": args.split,
                    "topk": args.topk,
                    "trained_epoch": checkpoint.get("epoch"),
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote per-image results to {args.save_report}")
    return 0


def report(records, args, checkpoint, dataset, model_classes, keep_columns, device):
    total = len(records)
    k = args.topk

    def pct(key):
        return 100.0 * sum(r[key] for r in records) / total

    hidden = [c for c, keep in zip(model_classes, keep_columns) if not keep]

    print(f"Checkpoint : {args.checkpoint} (trained through epoch {checkpoint.get('epoch')})")
    print(f"Manifest   : {args.manifest}  split={args.split}")
    print(f"Device     : {device}   Images: {total}")
    print(f"Model knows {len(model_classes)} classes; this split has {len(dataset.class_names)}.")
    if hidden:
        print(f"Not present here, hidden in 'restricted': {', '.join(hidden)}")

    print(f"\n=== Accuracy ({len(model_classes)} classes open / {int(keep_columns.sum())} restricted) ===")
    print(f"{'':<14}{'top-1':>9}{f'top-{k}':>9}")
    print(f"{'open':<14}{pct('correct_open_top1'):>8.2f}%{pct('correct_open_topk'):>8.2f}%")
    print(f"{'restricted':<14}{pct('correct_restricted_top1'):>8.2f}%{pct('correct_restricted_topk'):>8.2f}%")

    by_class = collections.defaultdict(list)
    for record in records:
        by_class[record["true"]].append(record)

    print(f"\n=== Per class (open) ===")
    print(f"{'class':<16}{'n':>4}{'top-1':>8}{f'top-{k}':>8}  most common wrong guess")
    for label in sorted(by_class):
        rows = by_class[label]
        n = len(rows)
        top1 = 100.0 * sum(r["correct_open_top1"] for r in rows) / n
        topk = 100.0 * sum(r["correct_open_topk"] for r in rows) / n
        wrong = collections.Counter(
            r["open_top1"] for r in rows if not r["correct_open_top1"]
        )
        note = f"{wrong.most_common(1)[0][0]} ({wrong.most_common(1)[0][1]})" if wrong else "-"
        print(f"{label:<16}{n:>4}{top1:>7.1f}%{topk:>7.1f}%  {note}")

    missed = [r for r in records if not r["correct_open_topk"]]
    if missed:
        print(f"\n=== Missed even in top-{k} ({len(missed)} of {total}) ===")
        worst = collections.Counter(r["true"] for r in missed)
        for label, count in worst.most_common(10):
            print(f"  {label:<16} {count}")

    confused = collections.Counter(
        (r["true"], r["open_top1"]) for r in records if not r["correct_open_top1"]
    )
    if confused:
        print("\n=== Most common confusions (true -> predicted) ===")
        for (truth, predicted), count in confused.most_common(10):
            print(f"  {truth:<16} -> {predicted:<16} {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on a manifest split. Never trains."
    )
    parser.add_argument(
        "--checkpoint",
        default=Path("models/resnet18_baseline/best_model.pt"),
        type=Path,
    )
    parser.add_argument(
        "--manifest", default=Path("data/processed/grocery_manifest.csv"), type=Path
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument(
        "--topk",
        default=3,
        type=int,
        help="Shortlist size. 3 matches the baseline training metrics.",
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--save-report", type=Path, help="Optional JSON file of per-image predictions."
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
