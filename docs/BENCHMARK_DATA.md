# Running a TruthLens detector benchmark

This is an offline experiment. It does not enable model inference in the Flask API and does not upload the dataset.

## 1. Prepare a labeled dataset

Keep image files out of Git unless you have a clear license and permission to redistribute them. Put the dataset in a local folder using this structure:

```text
dataset/
  real/
    camera/
      clean/
        photo-001.jpg
      jpeg/
        photo-002.jpg
    stock/
      clean/
        photo-003.png
  ai/
    generator-a/
      clean/
        generated-001.png
      jpeg/
        generated-002.jpg
    generator-b/
      clean/
        generated-003.webp
```

- `real/` and `ai/` are required labels.
- The first folder below each label identifies the real-image source or AI generator.
- The next folder identifies the transformation, such as `clean`, `jpeg`, `resize`, `crop`, or `screenshot`.
- Keep source identities, near-duplicates, and derived edits together when splitting calibration and test data. Never let the same source image or its transformed variants appear in both splits.
- Use licensed or consented images and retain dataset/source attribution outside the repository.

## 2. Install the optional benchmark detector

The detector is intentionally not a required TruthLens backend dependency. Choose a candidate, inspect its license and model-weight terms, then install it in a separate virtual environment. For the `lynote-ai/ai-image-detector` candidate, follow its current installation guide: https://github.com/lynote-ai/ai-image-detector

Do not assume that the candidate's advertised score is calibrated or reliable for your dataset.

## 3. Run inference and export scores

From the TruthLens repository root, with the optional `aidetector` package installed in the active environment:

```bash
python tools/run_detector_benchmark.py ./dataset --backend univfd --output ./benchmarks/univfd.csv
```

The script requires the detector result to explicitly expose `probability_ai`; it will fail rather than guessing another field. The output CSV records labels, scores, generator/source, and transformation.

## 4. Evaluate the scores

```bash
python tools/evaluate_detector.py ./benchmarks/univfd.csv --threshold 0.5
```

The report includes false positives on real images, AI recall, balanced accuracy, ROC AUC, and generator/transformation breakdowns.

## 5. Report honestly

- Tune thresholds only on a calibration split; report final results on an untouched test split.
- Include sample counts for each generator and transformation.
- Report false-positive rate on real images as a first-class safety metric.
- Include confidence intervals when sample sizes support them.
- Treat `probability_ai` as a model output, not a calibrated probability, unless calibration has been measured.
- A strong result on one benchmark does not establish generalization to unseen generators, social-media recompression, screenshots, edits, or adversarial examples.

Until a candidate passes this process, TruthLens should continue returning `Inconclusive` in the public analysis API.
