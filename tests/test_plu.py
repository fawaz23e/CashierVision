from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cashiervision.plu import load_plu_mapping, normalize_label, suggest_plu_codes


class PLUTest(unittest.TestCase):
    def test_normalize_label(self) -> None:
        self.assertEqual(normalize_label("Bell Pepper"), "bell_pepper")
        self.assertEqual(normalize_label("green-onion"), "green_onion")

    def test_banana_has_plu_suggestion(self) -> None:
        mapping = load_plu_mapping(PROJECT_ROOT / "data" / "plu_mapping.csv")
        suggestions = suggest_plu_codes("banana", mapping)
        self.assertTrue(any(record.plu_code == "4011" for record in suggestions))

    def test_unknown_class_returns_empty_list(self) -> None:
        mapping = load_plu_mapping(PROJECT_ROOT / "data" / "plu_mapping.csv")
        self.assertEqual(suggest_plu_codes("dragonfruit", mapping), [])


if __name__ == "__main__":
    unittest.main()

