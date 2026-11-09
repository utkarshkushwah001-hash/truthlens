# TruthLens detector shortlist

**Selected optional candidate: Smaug-AI-Image-Detector.** It is integrated as a model estimate behind an explicit local opt-in. TruthLens has not independently reproduced the model-card benchmark, so its output is not presented as proof or calibrated confidence.

## Selected — Smaug AI Image Detector

- Model card: https://huggingface.co/abacusai/Smaug-AI-Image-Detector
- Release announcement: https://blog.abacus.ai/blog/2026/10/17/smaug-ai-image-detector/
- Why selected: the publisher supplies local PyTorch weights and code, documents CPU and NVIDIA GPU operation, and reports evaluation against other detectors on 26,070 images from three public benchmarks, including a 5% false-alarm operating point.
- Caution: these figures are from the release author's evaluation. The project has not independently reproduced them; check license and current model limitations before deployment.

## Candidate B — lynote-ai/ai-image-detector

- Repository: https://github.com/lynote-ai/ai-image-detector
- License in repository metadata: MIT.
- Why test it: provides an installable CLI/API and benchmark commands, supports model backends including UnivFD and Sentry ConvNeXt, and documents limitations and reproducible evaluation workflows.
- Caution: its documented benchmark results are repository-maintainer results, not independent guarantees. It introduces PyTorch and computer-vision dependencies, so it remains a comparison candidate rather than the integrated model.

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
8. Keep the Smaug integration optional and retain an explicit `inconclusive` path when it is off or unavailable.

## Why the bar is high

A 2026 zero-shot benchmark covering 23 pretrained detector variants, 12 datasets, and 291 generators reports substantial variation in detector rankings and poor performance on some modern commercial generators. This argues against treating any single model-card metric as universal evidence.

Reference: https://arxiv.org/abs/2602.07814

