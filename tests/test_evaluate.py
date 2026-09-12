"""Tests for evaluating a trained checkpoint on a manifest with fewer classes.

The danger this guards against: the baseline model's class order and the grocery
manifest's class order agree for the first two entries and then diverge. Scoring
by raw integer index would compare eggplant against cauliflower and still print a
plausible-looking accuracy.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_on_manifest import evaluate

MODEL_CLASSES = ["banana", "cantaloupe", "cauliflower", "eggplant", "lemon"]
MANIFEST_CLASSES = ["banana", "cantaloupe", "eggplant", "lemon"]


def build_translation():
    model_class_to_idx = {name: i for i, name in enumerate(MODEL_CLASSES)}
    dataset_to_model = torch.tensor(
        [model_class_to_idx[name] for name in MANIFEST_CLASSES], dtype=torch.long
    )
    keep = torch.zeros(len(MODEL_CLASSES), dtype=torch.bool)
    keep[dataset_to_model] = True
    return dataset_to_model, keep


class ConstantModel(torch.nn.Module):
    """Returns fixed logits, so predictions are exactly what the test dictates."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.calls = 0

    def forward(self, images):
        start = self.calls
        self.calls += len(images)
        return self.logits[start : start + len(images)]


def run(logits, dataset_labels, k=3):
    dataset_to_model, keep = build_translation()
    model = ConstantModel(logits)
    loader = [(torch.zeros(len(dataset_labels), 3, 4, 4), torch.tensor(dataset_labels))]
    return evaluate(
        model, loader, torch.device("cpu"), dataset_to_model, keep, MODEL_CLASSES, k=k
    )


class TestEvaluate(unittest.TestCase):
    def test_translates_manifest_index_to_model_index(self):
        # Manifest index 2 is eggplant; model index 2 is cauliflower.
        # Predicting model index 3 (eggplant) must count as CORRECT.
        logits = torch.tensor([[0.0, 0.0, 0.0, 9.0, 0.0]])
        records = run(logits, [2])
        self.assertEqual(records[0]["true"], "eggplant")
        self.assertEqual(records[0]["open_top1"], "eggplant")
        self.assertTrue(records[0]["correct_open_top1"])

    def test_raw_index_match_is_not_counted_correct(self):
        # Predicting model index 2 (cauliflower) for manifest index 2 (eggplant)
        # is WRONG, even though the integers are equal.
        logits = torch.tensor([[0.0, 0.0, 9.0, 0.0, 0.0]])
        records = run(logits, [2])
        self.assertEqual(records[0]["true"], "eggplant")
        self.assertEqual(records[0]["open_top1"], "cauliflower")
        self.assertFalse(records[0]["correct_open_top1"])

    def test_restricted_hides_absent_classes(self):
        # cauliflower has the highest score but is absent from the manifest.
        # open picks cauliflower (wrong); restricted must fall back to eggplant.
        logits = torch.tensor([[0.0, 0.0, 9.0, 5.0, 0.0]])
        records = run(logits, [2])
        self.assertEqual(records[0]["open_top1"], "cauliflower")
        self.assertFalse(records[0]["correct_open_top1"])
        self.assertEqual(records[0]["restricted_top1"], "eggplant")
        self.assertTrue(records[0]["correct_restricted_top1"])

    def test_topk_uses_names_not_indices(self):
        logits = torch.tensor([[1.0, 0.5, 9.0, 8.0, 0.0]])
        records = run(logits, [2])
        self.assertEqual(records[0]["open_topk"], ["cauliflower", "eggplant", "banana"])
        self.assertTrue(records[0]["correct_open_topk"])

    def test_batch_of_mixed_labels(self):
        logits = torch.tensor(
            [
                [9.0, 0.0, 0.0, 0.0, 0.0],  # banana, manifest 0 -> correct
                [0.0, 0.0, 0.0, 0.0, 9.0],  # lemon,  manifest 3 -> correct
                [0.0, 9.0, 0.0, 0.0, 0.0],  # says cantaloupe for eggplant -> wrong
            ]
        )
        records = run(logits, [0, 3, 2])
        self.assertEqual([r["correct_open_top1"] for r in records], [True, True, False])
        self.assertEqual([r["true"] for r in records], ["banana", "lemon", "eggplant"])


    def test_restricted_topk_never_lists_absent_classes(self):
        # cauliflower is absent from the manifest, so it must not appear in the
        # restricted shortlist at any rank -- not just at rank 1.
        logits = torch.tensor([[1.0, 2.0, 9.0, 8.0, 0.5]])
        records = run(logits, [2])
        self.assertIn("cauliflower", records[0]["open_topk"])
        self.assertNotIn("cauliflower", records[0]["restricted_topk"])
        self.assertEqual(len(records[0]["restricted_topk"]), 3)

    def test_topk_is_clamped_to_available_classes(self):
        # Asking for top-10 from 5 model classes / 4 manifest classes must not
        # crash or pad with duplicates.
        logits = torch.tensor([[1.0, 2.0, 9.0, 8.0, 0.5]])
        records = run(logits, [2], k=10)
        self.assertEqual(len(records[0]["open_topk"]), 5)
        self.assertEqual(len(records[0]["restricted_topk"]), 4)
        self.assertEqual(len(set(records[0]["restricted_topk"])), 4)

    def test_top1_and_topk_disagree_when_truth_is_ranked_second(self):
        # Truth ranked 2nd: top-1 wrong, top-3 right. This is exactly the gap
        # a shortlist UI is meant to close.
        logits = torch.tensor([[0.0, 0.0, 9.0, 8.0, 0.0]])
        records = run(logits, [2])
        self.assertFalse(records[0]["correct_open_top1"])
        self.assertTrue(records[0]["correct_open_topk"])


if __name__ == "__main__":
    unittest.main()
