from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cashiervision.model import build_model
from cashiervision.training import freeze_backbone, run_epoch


class ScoreModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Identity()

    def forward(self, images):
        return self.fc(images)


class TrainingTest(unittest.TestCase):
    def test_training_updates_only_final_layer_and_preserves_batchnorm(self) -> None:
        torch.manual_seed(7)
        model = build_model(4, pretrained=False)
        freeze_backbone(model)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        loader = DataLoader(TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])), batch_size=2)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)
        run_epoch(model, loader, torch.device("cpu"), optimizer)

        self.assertFalse(torch.equal(before["fc.weight"], model.fc.weight))
        for name, value in model.state_dict().items():
            if not name.startswith("fc."):
                self.assertTrue(torch.equal(before[name], value), name)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("fc."))

        trained = {name: value.clone() for name, value in model.state_dict().items()}
        run_epoch(model, loader, torch.device("cpu"))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(trained[name], value), name)

    def test_metrics_weight_partial_batches_by_image_count(self) -> None:
        scores = torch.tensor([[4., 3., 2., 1.], [4., 3., 2., 1.], [1., 2., 3., 4.]])
        labels = torch.tensor([0, 2, 0])
        loader = DataLoader(TensorDataset(scores, labels), batch_size=2)
        metrics = run_epoch(ScoreModel(), loader, torch.device("cpu"))

        self.assertEqual(metrics["samples"], 3)
        self.assertAlmostEqual(metrics["top1_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["top3_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["loss"], nn.CrossEntropyLoss()(scores, labels).item(), places=6)
        limited = run_epoch(ScoreModel(), loader, torch.device("cpu"), max_batches=1)
        self.assertEqual(limited["samples"], 2)

    def test_two_classes_and_empty_loader(self) -> None:
        loader = DataLoader(TensorDataset(torch.tensor([[2., 1.]]), torch.tensor([1])))
        metrics = run_epoch(ScoreModel(), loader, torch.device("cpu"))
        self.assertEqual(metrics["top3_accuracy"], 1.0)
        empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, "empty"):
            run_epoch(ScoreModel(), empty, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
