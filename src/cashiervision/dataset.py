from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


VALID_SPLITS = {"train", "val", "test"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ManifestRow:
    filepath: Path
    label: str
    split: str


def read_manifest(manifest_path: Path) -> list[ManifestRow]:
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"filepath", "label", "split"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Manifest is missing required columns: {missing}")

        rows = [
            ManifestRow(
                filepath=Path(row["filepath"]),
                label=row["label"].strip(),
                split=row["split"].strip(),
            )
            for row in reader
        ]

    if not rows:
        raise ValueError("Manifest contains no image rows")

    invalid_splits = sorted({row.split for row in rows} - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Manifest contains invalid splits: {', '.join(invalid_splits)}")

    return rows


def build_image_transform(split: str, image_size: int = 224) -> Callable[[Image.Image], Tensor]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")

    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    resize_size = round(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ProduceDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        transform: Callable[[Image.Image], Tensor] | None = None,
        root_dir: Path | None = None,
        image_size: int = 224,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {split}")

        manifest_path = manifest_path.resolve()
        rows = read_manifest(manifest_path)
        self.root_dir = root_dir.resolve() if root_dir else manifest_path.parents[2]
        self.class_names = sorted({row.label for row in rows})
        self.class_to_idx = {label: index for index, label in enumerate(self.class_names)}
        self.transform = transform or build_image_transform(split, image_size)
        self.samples = [
            (self._resolve_path(row.filepath), self.class_to_idx[row.label])
            for row in rows
            if row.split == split
        ]

        if not self.samples:
            raise ValueError(f"Manifest contains no rows for split: {split}")

    def _resolve_path(self, image_path: Path) -> Path:
        return image_path if image_path.is_absolute() else self.root_dir / image_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path, label_index = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, label_index


def create_datasets(
    manifest_path: Path,
    root_dir: Path | None = None,
    image_size: int = 224,
) -> dict[str, ProduceDataset]:
    return {
        split: ProduceDataset(
            manifest_path=manifest_path,
            split=split,
            root_dir=root_dir,
            image_size=image_size,
        )
        for split in ("train", "val", "test")
    }
