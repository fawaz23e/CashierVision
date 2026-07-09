from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(input_dir: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        label = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((image_path, label))
    return rows


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    fraction = index / total
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + val_ratio:
        return "val"
    return "test"


def assign_splits_by_label(
    rows: list[tuple[Path, str]],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> list[tuple[Path, str, str]]:
    rng = random.Random(seed)
    rows_by_label: dict[str, list[Path]] = defaultdict(list)
    split_rows: list[tuple[Path, str, str]] = []

    for image_path, label in rows:
        rows_by_label[label].append(image_path)

    for label in sorted(rows_by_label):
        image_paths = rows_by_label[label][:]
        rng.shuffle(image_paths)
        for index, image_path in enumerate(image_paths):
            split_rows.append(
                (
                    image_path,
                    label,
                    split_name(index, len(image_paths), train_ratio, val_ratio),
                )
            )

    rng.shuffle(split_rows)
    return split_rows


def write_manifest(rows: list[tuple[Path, str]], output_path: Path, seed: int) -> None:
    split_rows = assign_splits_by_label(rows, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split"])
        writer.writeheader()
        for image_path, label, split in split_rows:
            writer.writerow(
                {
                    "filepath": image_path.as_posix(),
                    "label": label,
                    "split": split,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an image classification manifest.")
    parser.add_argument("--input", required=True, type=Path, help="ImageFolder-style input directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV manifest path.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for split reproducibility.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = iter_images(args.input)
    if not rows:
        print(f"No images found in {args.input}")
        return 1
    write_manifest(rows, args.output, args.seed)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
