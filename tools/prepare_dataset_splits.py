#!/usr/bin/env python3
"""Create deterministic group-aware train/calibration/test CSV splits.

Input CSV columns: path,label,group_id. Every related image (original,
near-duplicate, crop, resize, screenshot, or transformed variant) must share
one group_id. Groups are assigned whole to one split to prevent leakage.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "calibration", "test")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "group_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Manifest must contain columns: path,label,group_id")
        rows = []
        seen = set()
        for line, row in enumerate(reader, start=2):
            item = {key: (row.get(key) or "").strip() for key in required}
            if not all(item.values()):
                raise ValueError(f"Missing path, label, or group_id on CSV line {line}.")
            item["label"] = item["label"].lower()
            if item["label"] not in {"real", "ai"}:
                raise ValueError(f"Label on CSV line {line} must be 'real' or 'ai'.")
            if item["path"] in seen:
                raise ValueError(f"Duplicate path in manifest: {item['path']}")
            seen.add(item["path"])
            rows.append(item)
    if not rows:
        raise ValueError("Manifest contains no image rows.")
    if {row["label"] for row in rows} != {"real", "ai"}:
        raise ValueError("Manifest must contain both real and ai labels.")
    return rows


def split_groups(rows: list[dict[str, str]], ratios: tuple[float, float, float], seed: int) -> dict[str, list[dict[str, str]]]:
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Split ratios must all be > 0 and sum to 1.")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    if len(grouped) < 3:
        raise ValueError("At least 3 distinct groups are required for train/calibration/test.")
    groups = list(grouped.items())
    random.Random(seed).shuffle(groups)
    targets = [len(rows) * ratio for ratio in ratios]
    assigned: dict[str, list[dict[str, str]]] = {name: [] for name in SPLITS}
    counts = [0, 0, 0]
    # Greedy size balancing; group integrity takes priority over exact ratios.
    for group_id, members in sorted(groups, key=lambda pair: len(pair[1]), reverse=True):
        eligible = [i for i in range(3) if counts[i] < targets[i]]
        choices = eligible or list(range(3))
        idx = min(choices, key=lambda i: (counts[i] / targets[i], i))
        assigned[SPLITS[idx]].extend(members)
        counts[idx] += len(members)
    for split_name, split_rows in assigned.items():
        labels = {row["label"] for row in split_rows}
        if labels != {"real", "ai"}:
            raise ValueError(
                f"Split '{split_name}' does not contain both real and ai images. "
                "Add more groups or revise group assignments; the tool will not silently create an invalid evaluation split."
            )
    return assigned


def write_splits(splits: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        target = output_dir / f"{name}.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "group_id"])
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["path"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with path,label,group_id columns")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "CALIBRATION", "TEST"))
    args = parser.parse_args(argv)
    try:
        rows = read_manifest(args.manifest)
        splits = split_groups(rows, tuple(args.ratios), args.seed)
        write_splits(splits, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Split {len(rows)} images into:")
    for name in SPLITS:
        part = splits[name]
        print(f"  {name}: {len(part)} images, {len({row['group_id'] for row in part})} groups")
    print("Verify source/group IDs before trusting results; this tool cannot discover duplicates automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
