# TruthLens 🔍

TruthLens is an experimental web app for inspecting image files and, eventually, evaluating whether images may be AI-generated.

## Current status

The current backend validates supported image uploads and reports basic file metadata. **It does not currently detect AI-generated images or verify image provenance.** Valid uploads return an *Inconclusive* result rather than a fabricated authenticity score.

## Current features

- Image upload and local preview
- Backend image validation for JPEG, PNG, and WEBP
- Upload size and image-dimension limits
- Restricted local-development CORS configuration
- No persistent storage of uploaded images by the analysis endpoint
- Honest inconclusive result when no detector is configured

## Tech stack

- Frontend: HTML, Tailwind CSS, JavaScript
- Backend: Python, Flask, Flask-CORS, Pillow

## Run locally

From the repository root:

```bash
python -m pip install -r backend/requirements.txt
python backend/app.py
```

Serve the `frontend/` directory on port 5500 using a local static server. The frontend calls `http://127.0.0.1:5000/analyze` and the backend allows common local origins by default. For another local origin, set `TRUTHLENS_ALLOWED_ORIGINS` to a comma-separated list of exact origins.

Run backend tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Reproducible detector evaluation

Evaluate a saved score CSV with `label,score` columns (`real` and `ai` labels;
higher scores must have a documented AI-like meaning):

```powershell
python tools/evaluate_detector.py benchmarks/predictions.csv --backend univfd --split calibration --output benchmarks/report.json
python tools/calibrate_detector.py benchmarks/calibration.csv benchmarks/held_out.csv --objective balanced_accuracy --output benchmarks/calibrated-report.json
```

Reports include sample counts and class balance, accuracy, balanced accuracy,
precision, recall, F1, confusion matrix, ROC-AUC, and per-class raw-score
summaries. Group breakdowns are descriptive and may have undefined metrics if a
group has only one class. Choose thresholds from calibration data only; reserve
held-out data for a single final evaluation. These metrics do not establish
performance outside the evaluated dataset.

The local benchmark notes for UnivFD (200 CIFAKE images: 100 per label) report
ROC-AUC 0.3206 when higher is interpreted as more AI-like (0.6794 after
reversing direction), and 49.5% accuracy at 0.5. Sentry ConvNeXt Small reports
0.4340 AUC (0.5660 reversed) and 46.5% accuracy at 0.5. A separate 1,000-image
UnivFD run reports 0.2694 AUC in that same assumed direction. These preliminary
results do not support selecting either detector for user-facing classification.
The reverse-direction result is a diagnostic signal only: it can indicate a
label, adapter, preprocessing, checkpoint, or domain issue. Do not reverse
scores or tune a threshold after looking at held-out results.

CIFAKE images are 32×32 and do not represent the range of real uploads. Keep
dataset partitions and source groups separate; compare the manifest group IDs
and exact file hashes before making leakage claims. Detector outputs remain
experimental and the web application therefore continues to return
“Inconclusive.”

## Before adding an AI detector

A model must be evaluated on labeled, licensed or consented images, including held-out generator families and realistic image transformations. Measure false positives, false negatives, balanced accuracy, ROC AUC, and abstention coverage. Do not interpret raw model scores as calibrated probabilities without calibration evidence.

## Security and deployment notes

- The API limits uploads to 10 MB and decoded images to 40 megapixels.
- Uploads are processed in memory and are not written to disk.
- Debug mode is disabled. Do not expose Flask's development server to the public internet; use a production WSGI server for deployment.
- Configure CORS to the exact trusted frontend origin before deployment.
- AI-origin detection and provenance verification remain unconfigured.

