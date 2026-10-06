# Verification evidence

## Historical local verification — 5 October 2026

At this pre-publication checkpoint, real source archives were downloaded from the author mirror and SHA-256 verified. The full CNN and transfer-learning experiment ran on CPU. Six tests passed with the real trained artifact, including a valid API image and invalid base64 rejection; local CI-style runs without the artifact had five passing tests and one explicitly skipped real-artifact check. Ruff lint and format passed. The evidence notebook executed and recomputed macro-F1 from stored test predictions. Confusion matrix and Grad-CAM were inspected; the latter retains a confident error and a thumb/background activation.

A separate clean Python 3.12 environment was installed from pinned dependencies. A local Git clone downloaded all three archives and pretrained weights independently, then ran the complete training/calibration/test pipeline. Selected architecture, all validation/test/robustness metrics, per-image test probabilities and trained state tensors reproduce within atol=rtol=1e-6. No raw images, feature caches or model artifacts were copied into the clone. pip check passed. API TestClient emits an httpx deprecation warning; requests and assertions still pass.

Source ZIP excludes raw photos and model weights; these remain saved in the original local repository. Reproduction generates them. The repository has since been [published](https://github.com/idrisslemnouni-crypto/plant-disease-detection); see its [GitHub Actions history](https://github.com/idrisslemnouni-crypto/plant-disease-detection/actions) for remote results by revision. No independent-farm validation or production deployment is claimed.

## Input-contract verification — 6 October 2026

Inspected all 1,295 cached source photos: every EXIF orientation was 1 or missing and every file had one frame. Shared EXIF-oriented RGB decoding therefore preserves the original training inputs; no model was retrained or accuracy estimate revised. The same helper now serves inference, training tensors, image hashes, robustness views and Grad-CAM source images.

Ten local tests pass with the actual model, including asymmetric lossless EXIF-6/displayed-pixel equivalence through decoder/hash/training paths, multiframe API rejection before artifact loading, and inference batch-partition invariance for both architectures. Ruff lint/format and Git whitespace checks pass. The existing FastAPI/httpx deprecation warning remains. These are engineering contract checks on an already inspected study, without a new untouched-test or external-farm claim. They do not assert a remote CI result for the new changes; consult Actions for each pushed revision.
