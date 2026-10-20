# TruthLens detector shortlist

**Decision: do not expose a model score to users until TruthLens has reproduced a benchmark on a held-out dataset.** Vendor/model-card numbers are useful for choosing candidates, not proof of production reliability.

## Candidate A — Smaug AI Image Detector

- Model card: https://huggingface.co/abacusai/Smaug-AI-Image-Detector
- Release announcement: https://blog.abacus.ai/blog/2026/10/17/smaug-ai-image-detector/
- Why test it: very recent candidate; its release announcement reports an evaluation against 10 open-source detectors on 26,070 images from three public benchmarks, including an operating point at a 5% false-alarm rate.
- Caution: the reported figures are from the release author's evaluation. Independently reproduce results, inspect model license and inference requirements, and verify model-card limitations before integrating it.

## Candidate B — lynote-ai/ai-image-detector

- Repository: https://github.com/lynote-ai/ai-image-detector
- License in repository metadata: MIT.
- Why test it: provides an installable CLI/API and benchmark commands, supports model backends including UnivFD and Sentry ConvNeXt, and documents limitations and reproducible evaluation workflows.
- Caution: its documented benchmark results are repository-maintainer results, not independent guarantees. It introduces PyTorch and computer-vision dependencies; avoid adding the whole dependency stack to the lightweight Flask upload API until a candidate is selected.

## Candidate C — SigLIP2 + DINOv2 fine-tuned detector

- Model card: https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2
- Why test it: the model card reports evaluations across several real-image and generated-image datasets, including degraded images.
- Caution: the high headline metrics are model-card claims. Check the exact license, model size, hardware needs, training-data overlap, and whether test sets are independent before adoption.

## Required evaluation gate

1. Obtain a licensed/consented, labeled evaluation set with both real and generated images.
2. Keep source images and near-duplicates separated across calibration and test splits.
3. Include modern held-out generator families, including commercial generators where legally available.
4. Test compression, resizing, crops, screenshots, and edited images.
5. Report false-positive rate on real photographs, AI recall, balanced accuracy, ROC AUC, subgroup performance, inference latency, and abstention coverage.
6. Select thresholds on calibration data, then report once on an untouched test set.
7. Inspect model license and any custom-code requirement before deployment.
8. Only after these checks, implement a detector adapter behind an optional dependency and keep an explicit `inconclusive` path.

## Why the bar is high

A 2026 zero-shot benchmark covering 23 pretrained detector variants, 12 datasets, and 291 generators reports substantial variation in detector rankings and poor performance on some modern commercial generators. This argues against treating any single model-card metric as universal evidence.

Reference: https://arxiv.org/abs/2602.07814
