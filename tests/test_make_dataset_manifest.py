from collections import Counter, defaultdict
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from make_dataset_manifest import assign_splits_by_label, split_name


class MakeDatasetManifestTest(unittest.TestCase):
    def test_split_name_uses_expected_boundaries(self) -> None:
        self.assertEqual(split_name(0, 100, train_ratio=0.70, val_ratio=0.15), "train")
        self.assertEqual(split_name(70, 100, train_ratio=0.70, val_ratio=0.15), "val")
        self.assertEqual(split_name(85, 100, train_ratio=0.70, val_ratio=0.15), "test")

    def test_assign_splits_by_label_stratifies_each_class(self) -> None:
        rows = [
            (Path(f"data/raw/apple/{index}.jpg"), "apple")
            for index in range(20)
        ] + [
            (Path(f"data/raw/banana/{index}.jpg"), "banana")
            for index in range(20)
        ]

        split_rows = assign_splits_by_label(rows, seed=42)
        split_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
        for _image_path, label, split in split_rows:
            split_counts_by_label[label][split] += 1

        self.assertEqual(split_counts_by_label["apple"], {"train": 14, "val": 3, "test": 3})
        self.assertEqual(split_counts_by_label["banana"], {"train": 14, "val": 3, "test": 3})


if __name__ == "__main__":
    unittest.main()
