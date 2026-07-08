from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_dataset_manifest import audit_manifest


class ManifestAuditTest(unittest.TestCase):
    def test_valid_manifest_has_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            classes_path = tmp_path / "classes.txt"
            manifest_path = tmp_path / "manifest.csv"
            classes_path.write_text("apple\nbanana\n", encoding="utf-8")
            manifest_path.write_text(
                "\n".join(
                    [
                        "filepath,label,split",
                        "data/raw/imagefolder/apple/001.jpg,apple,train",
                        "data/raw/imagefolder/banana/001.jpg,banana,val",
                        "data/raw/imagefolder/banana/002.jpg,banana,test",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(audit_manifest(manifest_path, classes_path), [])

    def test_unexpected_label_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            classes_path = tmp_path / "classes.txt"
            manifest_path = tmp_path / "manifest.csv"
            classes_path.write_text("apple\nbanana\n", encoding="utf-8")
            manifest_path.write_text(
                "\n".join(
                    [
                        "filepath,label,split",
                        "data/raw/imagefolder/dragonfruit/001.jpg,dragonfruit,train",
                        "data/raw/imagefolder/apple/001.jpg,apple,val",
                        "data/raw/imagefolder/banana/001.jpg,banana,test",
                    ]
                ),
                encoding="utf-8",
            )

            errors = audit_manifest(manifest_path, classes_path)

        self.assertIn("Line 2: unexpected label 'dragonfruit'", errors)


if __name__ == "__main__":
    unittest.main()
