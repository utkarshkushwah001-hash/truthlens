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

## Before adding an AI detector

A model must be evaluated on labeled, licensed or consented images, including held-out generator families and realistic image transformations. Measure false positives, false negatives, balanced accuracy, ROC AUC, and abstention coverage. Do not interpret raw model scores as calibrated probabilities without calibration evidence.

## Security and deployment notes

- The API limits uploads to 10 MB and decoded images to 40 megapixels.
- Uploads are processed in memory and are not written to disk.
- Debug mode is disabled. Do not expose Flask's development server to the public internet; use a production WSGI server for deployment.
- Configure CORS to the exact trusted frontend origin before deployment.
- AI-origin detection and provenance verification remain unconfigured.
