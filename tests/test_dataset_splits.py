import csv
import tempfile
import unittest
from pathlib import Path

from tools.prepare_dataset_splits import read_manifest, split_groups, write_splits


class DatasetSplitTests(unittest.TestCase):
    def rows(self):
        rows = []
        for label in ("real", "ai"):
            for i in range(10):
                rows.append({"path": f"{label}/{i}.jpg", "label": label, "group_id": f"{label}-source-{i}"})
                rows.append({"path": f"{label}/{i}-crop.jpg", "label": label, "group_id": f"{label}-source-{i}"})
        return rows

    def test_deterministic_and_group_disjoint(self):
        rows = self.rows()
        first = split_groups(rows, (0.7, 0.15, 0.15), 42)
        self.assertEqual(first, split_groups(rows, (0.7, 0.15, 0.15), 42))
        seen = {}
        for split, members in first.items():
            self.assertEqual({row["label"] for row in members}, {"real", "ai"})
            for row in members:
                if row["group_id"] in seen:
                    self.assertEqual(seen[row["group_id"]], split)
                seen[row["group_id"]] = split

    def test_rejects_missing_columns_and_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            path.write_text("path,label\na.jpg,real\nb.jpg,ai\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "path,label,group_id"):
                read_manifest(path)
            path.write_text("path,label,group_id\na.jpg,real,g1\na.jpg,ai,g2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate path"):
                read_manifest(path)

    def test_writes_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "splits"
            write_splits(split_groups(self.rows(), (0.7, 0.15, 0.15), 3), output)
            for name in ("train", "calibration", "test"):
                with (output / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
                    self.assertIn("group_id", csv.DictReader(handle).fieldnames)


if __name__ == "__main__":
    unittest.main()
