from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cashiervision.model import build_model


class ProduceModelTest(unittest.TestCase):
    def test_batch_scores_support_classification_loss_and_backward(self) -> None:
        model = build_model(num_classes=18, pretrained=False)
        model.eval()
        scores = model(torch.zeros(2, 3, 64, 64))

        self.assertEqual(tuple(scores.shape), (2, 18))
        loss = torch.nn.CrossEntropyLoss()(scores, torch.tensor([0, 17]))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.fc.weight.grad)
        self.assertTrue(torch.isfinite(model.fc.weight.grad).all().item())

    def test_pretrained_default_requests_imagenet_weights(self) -> None:
        with patch("cashiervision.model.resnet18", return_value=resnet18(weights=None)) as factory:
            model = build_model(num_classes=18)

        factory.assert_called_once_with(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.assertEqual(model.fc.out_features, 18)

    def test_rejects_too_few_classes(self) -> None:
        for num_classes in (0, 1, -1):
            with self.subTest(num_classes=num_classes):
                with self.assertRaisesRegex(ValueError, "at least 2"):
                    build_model(num_classes, pretrained=False)


if __name__ == "__main__":
    unittest.main()
