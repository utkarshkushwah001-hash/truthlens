import csv
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "run_manifest_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_manifest_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def make_manifest(tmp_path: Path) -> Path:
    rows = []
    for split in ("train_pool", "official_test"):
        for label in ("real", "ai"):
            image = tmp_path / f"{split}_{label}.png"
            image.write_bytes(b"test")
            rows.append({
                "path": str(image),
                "label": label,
                "group_id": f"{split}:{label}",
                "official_split": split,
            })
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group_id", "official_split"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_read_manifest_filters_split_and_checks_files(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = MODULE.read_manifest(manifest, "official_test")
    assert len(rows) == 2
    assert {row["label"] for row in rows} == {"real", "ai"}
    assert {row["official_split"] for row in rows} == {"official_test"}


def test_score_rows_requires_probability_ai(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = MODULE.read_manifest(manifest, "train_pool")
    class BadDetector:
        def predict_path(self, path):
            return {"score": 0.9}
    try:
        MODULE.score_rows(rows, BadDetector())
    except ValueError as exc:
        assert "probability_ai" in str(exc)
    else:
        raise AssertionError("Expected missing probability_ai to fail")


def test_score_rows_accepts_valid_scores(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = MODULE.read_manifest(manifest, "train_pool")
    class Detector:
        def predict_path(self, path):
            return {"probability_ai": 0.75}
    scored = MODULE.score_rows(rows, Detector())
    assert len(scored) == 2
    assert all(row["score"] == 0.75 for row in scored)


def test_read_manifest_rejects_missing_paths(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group_id", "official_split"])
        writer.writeheader()
        writer.writerows([
            {"path": str(tmp_path / "missing_real.png"), "label": "real", "group_id": "r", "official_split": "train_pool"},
            {"path": str(tmp_path / "missing_ai.png"), "label": "ai", "group_id": "a", "official_split": "train_pool"},
        ])
    try:
        MODULE.read_manifest(manifest, "train_pool")
    except ValueError as exc:
        assert "do not exist" in str(exc)
    else:
        raise AssertionError("Expected missing paths to fail")
