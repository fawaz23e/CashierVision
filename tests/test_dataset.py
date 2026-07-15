from pathlib import Path
import csv
import sys
import tempfile
import unittest

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cashiervision.dataset import ProduceDataset, create_datasets


class ProduceDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest_path = self.root / "data" / "processed" / "manifest.csv"
        self.manifest_path.parent.mkdir(parents=True)

        rows = []
        for split, label, color in (
            ("train", "banana", "yellow"),
            ("val", "apple", "red"),
            ("test", "banana", "yellow"),
        ):
            image_path = self.root / "data" / "raw" / label / f"{split}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), color=color).save(image_path)
            rows.append(
                {
                    "filepath": image_path.relative_to(self.root).as_posix(),
                    "label": label,
                    "split": split,
                }
            )

        with self.manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["filepath", "label", "split"])
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_filters_rows_and_converts_label_to_index(self) -> None:
        dataset = ProduceDataset(self.manifest_path, split="val", image_size=64)

        image_tensor, label_index = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.class_names, ["apple", "banana"])
        self.assertEqual(label_index, dataset.class_to_idx["apple"])
        self.assertEqual(tuple(image_tensor.shape), (3, 64, 64))

    def test_all_splits_share_the_same_class_mapping(self) -> None:
        datasets = create_datasets(self.manifest_path, image_size=64)

        self.assertEqual(datasets["train"].class_to_idx, datasets["val"].class_to_idx)
        self.assertEqual(datasets["val"].class_to_idx, datasets["test"].class_to_idx)

    def test_rejects_unknown_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            ProduceDataset(self.manifest_path, split="development")


if __name__ == "__main__":
    unittest.main()
