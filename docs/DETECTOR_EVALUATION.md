# TruthLens detector evaluation protocol

TruthLens must not treat file size, dimensions, or EXIF metadata as evidence that an image is AI-generated. Before exposing any detector verdict, evaluate a candidate model on a labeled set that reflects intended use.

## Run the evaluation script

The evaluation tool uses only the Python standard library and expects scores produced by a detector on a held-out test set. It does not run inference.

CSV columns:

- `label`: `real` or `ai`
- `score`: finite value from 0 to 1, where higher means more likely AI-generated
- `generator` (optional): generator family/name for synthetic samples; use a consistent label for real samples such as `real`
- `transformation` (optional): e.g. `clean`, `jpeg_q70`, `resize`, `crop`

Example CSV:

```csv
label,score,generator,transformation
real,0.08,real,clean
real,0.31,real,jpeg_q70
ai,0.87,stable-diffusion,clean
ai,0.62,flux,resize
```

Run:

```bash
python tools/evaluate_detector.py path/to/scores.csv --threshold 0.5
```

The report includes confusion counts, AI recall/sensitivity, real-image specificity, AI precision, false-positive rate, balanced accuracy, ROC AUC, and breakdowns by generator and transformation. The threshold is an evaluation setting, not a calibrated probability cutoff.

## Minimum evaluation design

1. **Separate by source and generator.** Do not randomly split near-duplicates or variants of the same source image across train and test. Keep at least one modern generator family entirely held out.
2. **Balance the comparison.** Avoid letting image resolution, JPEG quality, watermarks, or dataset source reveal the label. Real and generated examples should have comparable content and processing.
3. **Test realistic transformations.** Evaluate clean originals, JPEG compression, resizing, crops, screenshots, and color changes. Keep the original label and record the transformation.
4. **Report false positives.** A false AI label on a real photograph can harm users. Review false positives separately by content type and source.
5. **Choose thresholds on validation data only.** Do not tune the threshold against the final test set. Report precision/recall trade-offs and the rate at which the detector abstains.
6. **Keep an untouched test set.** Publish sample counts, dataset provenance, licenses/consent, model revision, threshold, and reproducible evaluation code.
7. **Do not overclaim.** Detector scores are not calibrated probabilities unless calibration has been tested. A detector cannot reliably establish authorship or provenance from pixels alone.

## Current TruthLens status

No detector is integrated yet. The upload endpoint intentionally returns an inconclusive result. A candidate must be selected and tested against this protocol before the app exposes its scores to users.

## Why this matters

Recent large-scale zero-shot benchmarking reports that detector rankings vary considerably across datasets and generators, with some modern commercial generators defeating most evaluated models. Dataset artifacts such as JPEG compression and image size can also inflate reported performance. See:

- [Out-of-the-box benchmark of open-source AI-image detectors (2026)](https://arxiv.org/abs/2602.07814)
- [AI-GenBench (2025)](https://arxiv.org/abs/2504.20865)
- [Fake or JPEG? Dataset bias study (2024)](https://arxiv.org/abs/2403.17608)
