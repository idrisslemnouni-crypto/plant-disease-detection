# Design — Bean leaf classification

Use the Makerere/NaCRRI Beans field-photo dataset: 1,034 train, 133 validation, 128 test, three labels. Preserve supplied partitions, remove exact decoded-image duplicates with test > validation > train precedence, audit perceptual near matches and document unavailable plant/farm IDs. No independent geographic/domain validation is claimed.

Train a compact CNN from scratch with train-only augmentation as baseline; extract frozen ImageNet MobileNetV3 Small features with original/flip train views and train a PyTorch linear head. Select architecture/epoch on validation macro-F1, calibrate temperature on validation, evaluate test once. Report per-class errors, probability metrics and a predetermined blur/darkness robustness check, explicitly not external-domain testing. Grad-CAM visualizes the first test example per class, with both correct and incorrect predictions retained.

CPU-only fixed-seed workflow; raw images, feature caches and model weights stay ignored. Hash-pinned download manifest; standalone inference and strict FastAPI base64 endpoint. Notebook, evidence and French learning/interview guides. Local completion precedes daily GitHub publication. No agronomic diagnosis, broad disease detection or confident out-of-distribution rejection claim.

6 October input-contract improvement: a shared helper applies EXIF display orientation before RGB conversion in image hashes, training tensors, robustness views, Grad-CAM source images and inference. Multiframe images are rejected. All 1,295 cached source photos had orientation 1/missing and one frame, so the existing trained inputs/results are preserved. Lossless rotated-image equivalence and batch-partition invariance are contract tests, not new field validation.
