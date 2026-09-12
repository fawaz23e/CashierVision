import contextlib
import csv
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import train_model
from cashiervision.dataset import read_manifest
from cashiervision.training import save_checkpoint


class RandomDataset(Dataset):
    """Small stochastic inputs exercise shuffle and augmentation state restoration."""
    def __init__(self, manifest, split):
        rows = read_manifest(manifest)
        self.class_names = sorted({row.label for row in rows})
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        self.samples = [(row.filepath, self.class_to_idx[row.label]) for row in rows if row.split == split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.rand(4) + random.random(), self.samples[index][1]


class TinyModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, images):
        return self.fc(self.bn(images))


class ResumeTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.csv"
        self.classes = self.root / "classes.txt"
        names = ["apple", "banana", "lemon", "lime"]
        self.classes.write_text("\n".join(names), encoding="utf-8")
        with self.manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["filepath", "label", "split"])
            writer.writeheader()
            for split, count in (("train", 8), ("val", 5), ("test", 1)):
                for index in range(count):
                    writer.writerow({"filepath": f"{split}/{index}.jpg", "label": names[index % 4], "split": split})

    def run_training(self, output, epochs=1, resume=False, extra=()):
        args = ["train_model.py", "--manifest", str(self.manifest), "--classes", str(self.classes),
                "--device", "cpu", "--epochs", str(epochs)]
        args += ["--resume", str(output / "last_checkpoint.pt")] if resume else [
            "--output-dir", str(output), "--batch-size", "2",
        ]
        with patch.object(sys, "argv", args + list(extra)), \
                patch.object(train_model, "ProduceDataset", RandomDataset), \
                patch.object(train_model, "build_model", side_effect=TinyModel) as builder, \
                contextlib.redirect_stdout(io.StringIO()):
            train_model.main()
        self.assertEqual(builder.call_args.kwargs["pretrained"], not resume)
        return torch.load(output / "last_checkpoint.pt", weights_only=True)

    def assert_state_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_two_plus_two_epochs_match_four_uninterrupted_epochs(self):
        full = self.run_training(self.root / "full", epochs=4)
        resumed_dir = self.root / "resumed"
        self.run_training(resumed_dir, epochs=2)
        resumed = self.run_training(resumed_dir, epochs=2, resume=True)
        self.assertEqual(resumed["epoch"], 4)
        for name in ("model_state_dict", "optimizer_state_dict", "history", "rng_state"):
            self.assert_state_equal(full[name], resumed[name])
        self.assert_state_equal(full["best_model"]["model_state_dict"], resumed["best_model"]["model_state_dict"])
        self.assertEqual(full["best_model"]["epoch"], resumed["best_model"]["epoch"])

    def test_resume_preserves_better_previous_epoch_and_recovers_best_export(self):
        output = self.root / "best"
        first = self.run_training(output)
        first["best_model"]["validation"]["top1_accuracy"] = 1.0
        save_checkpoint(first, output / "last_checkpoint.pt")
        (output / "best_model.pt").unlink()
        resumed = self.run_training(output, resume=True)
        self.assertEqual(resumed["best_model"]["epoch"], 1)
        best = torch.load(output / "best_model.pt", weights_only=True)
        self.assert_state_equal(best, first["best_model"])

    def test_changed_manifest_or_settings_do_not_overwrite_checkpoint(self):
        output = self.root / "validation"
        self.run_training(output)
        original = (output / "last_checkpoint.pt").read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_training(output, resume=True, extra=("--batch-size", "3"))
        with self.manifest.open("a", encoding="utf-8") as file:
            file.write("\n")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_training(output, resume=True)
        self.assertEqual(original, (output / "last_checkpoint.pt").read_bytes())

    def test_failed_save_keeps_previous_checkpoint(self):
        path = self.root / "checkpoint.pt"
        save_checkpoint({"epoch": 1}, path)
        def fail_save(checkpoint, file):
            file.write(b"partial checkpoint")
            raise OSError("disk full")
        with patch("cashiervision.training.torch.save", side_effect=fail_save):
            with self.assertRaisesRegex(OSError, "disk full"):
                save_checkpoint({"epoch": 2}, path)
        self.assertEqual(torch.load(path, weights_only=True), {"epoch": 1})


if __name__ == "__main__":
    unittest.main()
