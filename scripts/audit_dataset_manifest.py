from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = {"filepath", "label", "split"}
EXPECTED_SPLITS = {"train", "val", "test"}


def load_class_list(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }


def audit_manifest(manifest_path: Path, class_list_path: Path) -> list[str]:
    errors: list[str] = []
    valid_classes = load_class_list(class_list_path)
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            return [f"Missing columns: {sorted(missing_columns)}"]

        for line_number, row in enumerate(reader, start=2):
            filepath = row["filepath"].strip()
            label = row["label"].strip()
            split = row["split"].strip()

            if not filepath:
                errors.append(f"Line {line_number}: filepath is blank")
            if label not in valid_classes:
                errors.append(f"Line {line_number}: unexpected label '{label}'")
            if split not in EXPECTED_SPLITS:
                errors.append(f"Line {line_number}: unexpected split '{split}'")

            label_counts[label] += 1
            split_counts[split] += 1

    if not label_counts:
        errors.append("Manifest has no rows")

    missing_splits = EXPECTED_SPLITS - set(split_counts)
    if missing_splits:
        errors.append(f"Missing splits: {sorted(missing_splits)}")

    return errors


def summarize_manifest(manifest_path: Path) -> tuple[Counter[str], Counter[str]]:
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_counts[row["label"].strip()] += 1
            split_counts[row["split"].strip()] += 1

    return label_counts, split_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a CashierVision dataset manifest.")
    parser.add_argument(
        "--manifest",
        default=Path("data/processed/manifest.csv"),
        type=Path,
        help="Manifest CSV with filepath, label, and split columns.",
    )
    parser.add_argument(
        "--classes",
        default=Path("configs/classes_mvp.txt"),
        type=Path,
        help="Approved class list.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = audit_manifest(args.manifest, args.classes)
    if errors:
        for error in errors:
            print(error)
        return 1

    label_counts, split_counts = summarize_manifest(args.manifest)
    print(f"Manifest is valid: {args.manifest}")
    print("\nSplit counts:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    print("\nLabel counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
