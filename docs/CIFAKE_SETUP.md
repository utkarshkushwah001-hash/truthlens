# CIFAKE local smoke-test setup

1. Open the [CIFAKE dataset repository](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) and follow its dataset download instructions.
2. Download/extract the dataset locally. Do not commit image files or the downloaded archive to TruthLens.
3. Locate the extracted root containing `TRAIN/REAL`, `TRAIN/FAKE`, `TEST/REAL`, and `TEST/FAKE` (case-insensitive).
4. From the TruthLens repository root, run:

```powershell
python tools/prepare_cifake_manifest.py "D:\TruthLensData\CIFAKE" --output .\benchmarks\cifake_manifest.csv
```

Replace the example path with the actual extracted folder. The tool writes absolute image paths, labels, stable group IDs, and an `official_split` column. It does not download or copy images.

- `train_pool` contains the official training partition; create a separate calibration split within it before threshold selection.
- `official_test` remains untouched for final evaluation. Do not feed the whole manifest into the generic random splitter, because that would mix the official test set back into training/calibration.
- The manifest helper only checks the directory layout and image extensions; it does not decode or validate every image.
- CIFAKE images are 32x32 and therefore only a pipeline smoke test. Results do not establish performance on high-resolution user uploads.
- Review the dataset license and source-image rights for your intended use. Do not assume free download grants commercial redistribution rights.

After creating the manifest, inspect its printed counts and CSV paths before running any model.
