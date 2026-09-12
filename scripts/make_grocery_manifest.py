"""Build a CashierVision manifest from the Grocery Store Dataset.

Unlike scripts/make_dataset_manifest.py, this script does NOT invent its own
random split. The Grocery Store Dataset ships train.txt / val.txt / test.txt,
and we reuse those memberships so results stay comparable to the paper.

Two things this script does do:

1. Keeps only the classes listed in configs/grocery_class_map.csv, renaming
   them to our label names (for example Aubergine -> eggplant).
2. Tops up validation for classes the upstream val split forgot, by moving a
   few images out of train. The test split is never touched.

Rows are matched by the dataset's fine class ID, not by parsing folder names,
because folder depth differs between classes (Fruit/Banana/... has one level,
Fruit/Apple/Royal-Gala/... has two).
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

# These names match both the dataset's list files and the split column our
# manifests already use, so no translation is needed.
SPLITS = ("train", "val", "test")


def load_class_map(path: Path) -> dict[str, str]:
    """Return {grocery_class_name: our_label} from the explicit mapping file."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"grocery_class", "cashiervision_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        return {
            row["grocery_class"].strip(): row["cashiervision_label"].strip()
            for row in reader
        }


def load_fine_ids(classes_csv: Path) -> dict[int, str]:
    """Return {fine_class_id: grocery_class_name} from the dataset's classes.csv."""
    with classes_csv.open(newline="", encoding="utf-8") as file:
        return {
            int(row["Class ID (int)"]): row["Class Name (str)"].strip()
            for row in csv.DictReader(file)
        }


def read_split(split_txt: Path, fine_ids: dict[int, str]) -> list[tuple[str, str]]:
    """Return [(relative_path, grocery_class_name)] for one split list file."""
    rows = []
    for line_number, line in enumerate(
        split_txt.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"{split_txt}:{line_number}: expected 3 fields, got {len(parts)}")
        relative_path, fine_id, _coarse_id = parts
        try:
            name = fine_ids[int(fine_id)]
        except KeyError:
            raise SystemExit(f"{split_txt}:{line_number}: unknown class id {fine_id}") from None
        rows.append((relative_path, name))
    return rows


def main() -> int:
    args = parse_args()

    # Resolve both so relative_to() below produces clean repo-relative paths,
    # matching the convention in data/processed/manifest.csv.
    dataset_dir: Path = args.dataset_dir.resolve()
    repo_root = Path.cwd().resolve()

    class_map = load_class_map(args.class_map)
    fine_ids = load_fine_ids(dataset_dir / "classes.csv")

    unknown = sorted(set(class_map) - set(fine_ids.values()))
    if unknown:
        raise SystemExit(
            f"{args.class_map}: these grocery_class names are not in classes.csv: {unknown}"
        )

    # entries[split] = [(repo_relative_path, our_label)]
    entries: dict[str, list[tuple[str, str]]] = {}
    for split in SPLITS:
        kept = []
        for relative_path, name in read_split(dataset_dir / f"{split}.txt", fine_ids):
            label = class_map.get(name)
            if label is None:
                continue  # A class we are not using (apples, onions, packages...).
            absolute = dataset_dir / relative_path
            if not absolute.is_file():
                raise SystemExit(f"Listed image does not exist on disk: {absolute}")
            kept.append((str(absolute.relative_to(repo_root)), label))
        entries[split] = kept

    moved = top_up_validation(entries, args.min_val_per_class)

    write_manifest(args.output, entries, moved)
    report(args.output, entries, moved, args.min_val_per_class)
    return 0


def top_up_validation(
    entries: dict[str, list[tuple[str, str]]], minimum: int
) -> set[str]:
    """Move train images into val for classes short on validation images.

    Returns the set of moved filepaths so the manifest can record them.

    We take a contiguous block from the END of each class's train list. The
    dataset often contains several photos of one physical item, and those tend
    to sit next to each other in the numbering. Taking a block keeps them
    together instead of scattering the same item across train and val. This is
    an assumption about the numbering, NOT a verified fact -- the filenames
    restart at 001 in every split, so there is no item ID to group on.
    """
    val_counts = collections.Counter(label for _, label in entries["val"])
    train_by_label: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for row in entries["train"]:
        train_by_label[row[1]].append(row)

    moved: set[str] = set()
    for label, rows in sorted(train_by_label.items()):
        shortfall = minimum - val_counts[label]
        if shortfall <= 0:
            continue
        if shortfall >= len(rows):
            raise SystemExit(
                f"{label}: cannot move {shortfall} of only {len(rows)} train images."
            )
        block = sorted(rows)[-shortfall:]
        moved.update(path for path, _ in block)
        entries["val"].extend(block)

    entries["train"] = [row for row in entries["train"] if row[0] not in moved]
    entries["val"].sort()
    entries["train"].sort()
    return moved


def write_manifest(
    output: Path, entries: dict[str, list[tuple[str, str]]], moved: set[str]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filepath", "label", "split", "split_source"])
        for split in SPLITS:
            for filepath, label in entries[split]:
                source = "moved_from_train" if filepath in moved else "upstream"
                writer.writerow([filepath, label, split, source])


def report(
    output: Path,
    entries: dict[str, list[tuple[str, str]]],
    moved: set[str],
    minimum: int,
) -> None:
    counts: dict[str, collections.Counter] = {
        split: collections.Counter(label for _, label in entries[split]) for split in SPLITS
    }
    labels = sorted({label for split in SPLITS for label in counts[split]})

    print(f"Wrote {output}")
    print(f"\n{'label':<16}{'train':>7}{'val':>6}{'test':>7}")
    for label in labels:
        print(
            f"{label:<16}{counts['train'][label]:>7}"
            f"{counts['val'][label]:>6}{counts['test'][label]:>7}"
        )
    totals = [len(entries[split]) for split in SPLITS]
    print(f"{'TOTAL':<16}{totals[0]:>7}{totals[1]:>6}{totals[2]:>7}")
    print(f"\nClasses: {len(labels)}   Images: {sum(totals)}")

    if moved:
        print(
            f"\nMoved {len(moved)} image(s) from train to validation so every class "
            f"has at least {minimum} validation images."
            "\nThese are marked split_source=moved_from_train in the manifest."
            "\nThe official test split was not modified."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CashierVision manifest from the Grocery Store Dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default=Path("data/GroceryStoreDataset-master/dataset"),
        type=Path,
        help="Directory holding classes.csv and the train/val/test list files.",
    )
    parser.add_argument(
        "--class-map",
        default=Path("configs/grocery_class_map.csv"),
        type=Path,
        help="Explicit grocery_class -> cashiervision_label mapping.",
    )
    parser.add_argument(
        "--output",
        default=Path("data/processed/grocery_manifest.csv"),
        type=Path,
        help="Manifest CSV to write.",
    )
    parser.add_argument(
        "--min-val-per-class",
        default=4,
        type=int,
        help="Move train images into validation for classes below this count.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
