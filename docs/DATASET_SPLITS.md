# Group-aware dataset splits

Before benchmarking, create a CSV manifest with these columns:

```csv
path,label,group_id
dataset/real/camera/clean/photo-001.jpg,real,camera-roll-2025-001
dataset/real/camera/jpeg/photo-001.jpg,real,camera-roll-2025-001
dataset/ai/model-a/clean/generated-001.png,ai,model-a-prompt-001
dataset/ai/model-a/jpeg/generated-001.jpg,ai,model-a-prompt-001
```

**Important:** assign the same `group_id` to an original image and every crop, resize, recompressed copy, screenshot, or other derivative. Use a group ID for a source session or prompt family when appropriate. The script does not inspect pixels or automatically find near-duplicates; incorrect group IDs can still cause leakage.

From the TruthLens repository root:

```powershell
python tools/prepare_dataset_splits.py .\dataset_manifest.csv --output-dir .\benchmarks\splits --seed 42
```

This creates `train.csv`, `calibration.csv`, and `test.csv` with a default 70/15/15 target. Groups are never split across partitions. Actual ratios may differ because groups are indivisible. The script fails if any partition does not contain both `real` and `ai`, or if fewer than three groups are provided.

- Fit or choose model settings using training data only.
- Select a decision threshold on `calibration.csv` only.
- Report final metrics once on the untouched `test.csv`.
- Check that generator families and source identities are appropriate for the intended generalization claim; group splitting alone does not guarantee a representative benchmark.

The split CSVs list paths and labels; they do not copy images or download datasets. Review dataset licensing and attribution before use.
