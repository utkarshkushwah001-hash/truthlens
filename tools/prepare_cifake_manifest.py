#!/usr/bin/env python3
"""Build a manifest from CIFAKE's official TRAIN/TEST REAL/FAKE folders.

Expected layout (case-insensitive folder names):
  CIFAKE_ROOT/
    train/real/...
    train/fake/...
    test/real/...
    test/fake/...

The official test partition is labeled as 'official_test' and is not passed
through the random group splitter. The training partition is labeled
'train_pool' for a later train/calibration split. No images are copied.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_child(parent: Path, name: str) -> Path | None:
    if not parent.is_dir():
        return None
    return next((child for child in parent.iterdir() if child.is_dir() and child.name.lower() == name), None)


def resolve_layout(root: Path) -> dict[str, Path]:
    # Common CIFAKE variants use TRAIN/TEST or train/test.
    result = {}
    for split in ("train", "test"):
        split_dir = find_child(root, split)
        if split_dir is None:
            raise ValueError(f"Could not find {split}/ under {root}. Check the extracted CIFAKE folder.")
        real_dir = find_child(split_dir, "real")
        fake_dir = find_child(split_dir, "fake")
        if real_dir is None or fake_dir is None:
            raise ValueError(f"Expected {split}/REAL and {split}/FAKE folders under {split_dir}.")
        result[f"{split}_real"] = real_dir
        result[f"{split}_ai"] = fake_dir
    return result


def build_rows(root: Path) -> list[dict[str, str]]:
    folders = resolve_layout(root)
    rows = []
    for key, folder in folders.items():
        label = "real" if key.endswith("_real") else "ai"
        official_split = "official_test" if key.startswith("test_") else "train_pool"
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"No supported image files found in {folder}.")
        for path in files:
            # CIFAKE files are treated as independent examples. If you create
            # transformed copies later, give those copies the same group_id.
            rel = path.relative_to(root).as_posix()
            rows.append({
                "path": str(path.resolve()),
                "label": label,
                "group_id": f"cifake:{rel}",
                "official_split": official_split,
            })
    for split in ("train_pool", "official_test"):
        labels = {row["label"] for row in rows if row["official_split"] == split}
        if labels != {"real", "ai"}:
            raise ValueError(f"{split} must contain both REAL and FAKE images.")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cifake_root", type=Path, help="Extracted CIFAKE root directory")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/cifake_manifest.csv"))
    args = parser.parse_args(argv)
    try:
        rows = build_rows(args.cifake_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "group_id", "official_split"])
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} rows to {args.output}")
    for split in ("train_pool", "official_test"):
        for label in ("real", "ai"):
            count = sum(row["official_split"] == split and row["label"] == label for row in rows)
            print(f"  {split}/{label}: {count}")
    print("Keep official_test untouched. Split train_pool separately for training/calibration.")
    print("CIFAKE's 32x32 images are a pipeline smoke test, not proof of high-resolution performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
