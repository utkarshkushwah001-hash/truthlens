import csv
import tempfile
import unittest
from pathlib import Path

from tools.prepare_cifake_manifest import build_rows, resolve_layout


class CIFAKEMManifestTests(unittest.TestCase):
    def make_layout(self, root):
        for split in ("TRAIN", "TEST"):
            for label in ("REAL", "FAKE"):
                folder = root / split / label
                folder.mkdir(parents=True)
                (folder / f"{split.lower()}-{label.lower()}.png").write_bytes(b"not decoded by manifest tool")

    def test_builds_labels_and_preserves_official_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_layout(root)
            rows = build_rows(root)
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["official_split"] for row in rows}, {"train_pool", "official_test"})
            self.assertEqual({row["label"] for row in rows}, {"real", "ai"})
            self.assertEqual(len({row["group_id"] for row in rows}), 4)

    def test_fails_for_wrong_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            with self.assertRaisesRegex(ValueError, "Could not find train"):
                resolve_layout(root)

    def test_ignores_non_image_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_layout(root)
            (root / "TRAIN" / "REAL" / "notes.txt").write_text("ignore", encoding="utf-8")
            rows = build_rows(root)
            self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
