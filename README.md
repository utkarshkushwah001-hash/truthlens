# TruthLens

TruthLens is a lightweight image validation web application with an offline evaluation toolkit for AI image detector research.

## Current capability and detector status

The web application validates JPEG, PNG, and WEBP uploads and can run the optional local Smaug AI-image detector. When enabled, it returns **Likely AI-generated** or **Likely real** with a raw model score. The score is not calibrated confidence or proof of origin. Image edits and unfamiliar generators can still be misclassified.

Existing CIFAKE experiments are retained in `benchmarks/`. On the reported samples, UnivFD AUC was 0.3206 for 200 images and 0.2694 for a separate 1,000-image sample in the documented higher-means-AI direction. Sentry ConvNeXt Small AUC was 0.4340 for 200 images. These preliminary CIFAKE results do not support production classification. Reversed AUC is diagnostic evidence only; do not flip scores or tune a final test threshold to improve these numbers. CIFAKE's 32×32 images are a pipeline benchmark, not a representative sample of ordinary high-resolution uploads.

## Features

- Responsive image upload with selection, drag-and-drop, local preview, and upload progress.
- In-memory validation of MIME type, filename extension, decoded image format, file size, and pixel dimensions.
- Consistent JSON errors, restricted CORS, and no upload persistence.
- Optional local Smaug inference with an explicit unavailable and inconclusive state when it is disabled or cannot load.
- Offline tools for detector benchmarking, score evaluation, calibration, and leakage-aware dataset splits.

## Technology

- Python 3.12, Flask, Flask-CORS, Pillow, python-dotenv, Waitress.
- HTML, Tailwind CSS CDN, and vanilla JavaScript.
- Standard-library `unittest` suite.

## Repository layout

```text
backend/       Flask API, configuration, and Waitress entry point
frontend/      Static pages, styles, and upload workflow
tests/         API, benchmark, metric, and split tests
tools/         Offline benchmark and evaluation commands
benchmarks/    Existing score CSVs and dataset split manifests
docs/          Evaluation, security, dataset, and architecture notes
```

## Windows setup

Run from the project root in PowerShell. These commands use the venv interpreter directly, so PowerShell script activation policy does not need to change.

```powershell
Set-Location 'C:\Users\utkar\Downloads\truthlens\truthlens-main'
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` only when the frontend origin or port changes. `.env` is ignored by Git; `.env.example` contains local, non-secret defaults.

The detector runtime is optional because it installs PyTorch and downloads roughly 910 MB of model files on first inference. To enable it, install the optional runtime and ensure the generated `.env` contains `TRUTHLENS_DETECTOR=smaug`:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r backend\detector-requirements.txt
```

The detector runs locally on CPU or a supported NVIDIA GPU; uploaded images are not sent to a model API. The first inference needs internet access to download the pinned model revision and enough free disk space. Its weights use an Apache 2.0 license. The publisher reports strong results on OpenFake and CommunityForensics datasets, but those are publisher-run results, not independent validation for this project's users.

## Run locally

Start the Flask development server in one PowerShell window:

```powershell
Set-Location 'C:\Users\utkar\Downloads\truthlens\truthlens-main'
& .\.venv\Scripts\python.exe backend\app.py
```

Serve the frontend in a second window:

```powershell
Set-Location 'C:\Users\utkar\Downloads\truthlens\truthlens-main'
py -3.12 -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500`. The API health endpoint is `http://127.0.0.1:5000/health`.

The API defaults to these allowed browser origins:

```text
http://127.0.0.1:5500
http://localhost:5500
```

Set `TRUTHLENS_ALLOWED_ORIGINS` in `.env` to a comma-separated list of exact `http://` or `https://` origins. Wildcards, paths, and credentials in origins are rejected. When deploying the frontend, also set `frontend/js/config.js` `apiUrl` to the public API URL. Do not put secrets in frontend configuration.

## Tests

```powershell
Set-Location 'C:\Users\utkar\Downloads\truthlens\truthlens-main'
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## API behavior

`POST /analyze` accepts one multipart file in the `image` field. With `TRUTHLENS_DETECTOR=smaug` and the optional runtime installed, valid files receive HTTP 200 with a `model_estimate` result, verdict, and raw AI score. When detection is disabled, valid files return `inconclusive`. Missing runtime/model errors return HTTP 503. Unsupported format or mismatched MIME/extension receives HTTP 415; malformed image data receives HTTP 400; oversized requests or images receive HTTP 413. API errors are JSON. Files are decoded and scored in memory; uploads are not written to `backend/uploads/` or sent to a third-party inference service.

Limits are 10 MB per request and 40 megapixels per image. The client-side checks improve usability; the backend independently validates all upload data.

## Reproducible detector evaluation

Existing evaluation and calibration scripts accept saved scores; they do not run model inference or validate a model by themselves:

```powershell
& .\.venv\Scripts\python.exe tools\evaluate_detector.py benchmarks\predictions.csv --backend univfd --split calibration --output benchmarks\report.json
& .\.venv\Scripts\python.exe tools\calibrate_detector.py benchmarks\calibration.csv benchmarks\held_out.csv --objective balanced_accuracy --output benchmarks\calibrated-report.json
```

Keep source groups separate across training, calibration, and final test data. Choose thresholds on calibration data only and evaluate once on an untouched final test set. Report class balance, accuracy, balanced accuracy, precision, recall, F1, confusion matrix, ROC-AUC, and false-positive rate. Detector outputs are not calibrated confidence unless calibration has been independently demonstrated.

See [detector evaluation](docs/DETECTOR_EVALUATION.md), [dataset splits](docs/DATASET_SPLITS.md), and [benchmark setup](docs/BENCHMARK_DATA.md) before running new experiments. The original score CSVs and manifests are preserved.

## Deployment preparation

For a Windows-hosted API, Waitress is included as a production WSGI server. Start it behind a TLS-terminating reverse proxy, and keep it bound to loopback when the proxy runs on the same machine:

```powershell
Set-Location 'C:\Users\utkar\Downloads\truthlens\truthlens-main'
& .\.venv\Scripts\python.exe backend\serve.py
```

Set `TRUTHLENS_ALLOWED_ORIGINS` to the deployed frontend origin. Serve the static frontend over HTTPS and set its `apiUrl` to the HTTPS API endpoint. Add process supervision, log rotation, monitoring, and a reverse proxy before public exposure. No production service has been deployed.

## Security and limitations

- The backend listens on `127.0.0.1` by default and does not enable Flask debug mode.
- Uploads are never saved. `backend/uploads/` is Git-ignored to protect legacy/local files.
- CORS is restricted to exact configured origins.
- Use HTTPS, a reverse proxy, and suitable operational controls before public exposure.
- Smaug's model score is an estimate, not calibrated probability or proof of authorship/provenance. Model rankings vary across datasets and new generators may evade detectors.
- A detector that performs well on one dataset can fail on unseen generators, edits, resizing, screenshots, or recompression. Do not use the result as the sole basis for decisions about a person or their work.

