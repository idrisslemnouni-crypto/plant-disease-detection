import base64
import io

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import app.api as api
from plantvision.data import dhash, pixel_hash
from plantvision.inference import decode_image
from plantvision.models import build, gradcam
from plantvision.train import logits, probabilities, tensors


def test_softmax_temperature_contract():
    p = probabilities(np.array([[1, 2, 3.0]]), 2.0)
    np.testing.assert_allclose(p.sum(1), 1)
    assert p.argmax() == 2
    with pytest.raises(ValueError):
        probabilities([[1, 2, 3]], 0)


def test_decoded_pixel_hash_independent_of_encoding():
    image = Image.new("RGB", (32, 32), "green")
    assert pixel_hash(image) == pixel_hash(image.copy())


def test_invalid_images_and_dimensions():
    with pytest.raises(ValueError):
        decode_image(b"bad")
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    with pytest.raises(ValueError):
        decode_image(buffer.getvalue())


def test_exif_orientation_matches_display_pixels_and_training(tmp_path):
    pixels = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = 6  # Display orientation: 90 degrees clockwise.
    oriented = tmp_path / "oriented.png"
    displayed = tmp_path / "displayed.png"
    image.save(oriented, exif=exif)
    expected = image.transpose(Image.Transpose.ROTATE_270)
    expected.save(displayed)
    decoded = decode_image(oriented.read_bytes())
    np.testing.assert_array_equal(np.asarray(decoded), np.asarray(expected))
    assert decoded.getexif().get(274) is None
    with Image.open(oriented) as tagged:
        assert pixel_hash(tagged) == pixel_hash(expected)
        assert dhash(tagged) == dhash(expected)
    frame = pd.DataFrame({"path": [oriented.name, displayed.name], "label": [0, 0]})
    x, _ = tensors(tmp_path, frame, 32)
    torch.testing.assert_close(x[0], x[1], rtol=0, atol=0)


def test_multiframe_upload_rejected_before_model_loading(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=[Image.new("RGB", (32, 32), "blue")],
    )
    content = buffer.getvalue()
    with pytest.raises(ValueError, match="Multiframe"):
        decode_image(content)
    artifact = tmp_path / "unused.pt"
    artifact.write_bytes(b"Decoder must reject before this artifact is loaded")
    monkeypatch.setattr(api, "ARTIFACT", artifact)
    response = TestClient(api.app).post(
        "/predict", json={"image_base64": base64.b64encode(content).decode()}
    )
    assert response.status_code == 422
    assert "Multiframe" in response.json()["detail"]


@pytest.mark.parametrize("kind", ["small_cnn", "mobilenet_transfer"])
def test_inference_logits_do_not_depend_on_batch_partition(kind):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build(kind)
    model.train()  # Inference must disable dropout and batch-statistic updates.
    x = torch.randn((5, 3, 32, 32))
    np.testing.assert_allclose(logits(model, x, batch=1), logits(model, x, batch=3), atol=1e-6)
    assert not model.training


def test_cnn_output_and_gradcam():
    torch.set_num_threads(2)
    model = build("small_cnn")
    x = torch.ones((1, 3, 32, 32))
    assert model(x).shape == (1, 3)
    cam = gradcam(model, x, 0)
    assert cam.ndim == 2 and np.isfinite(cam).all()
    assert 0 <= cam.min() <= cam.max() <= 1


def test_api_missing_artifact_and_strict_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ARTIFACT", tmp_path / "absent.pt")
    client = TestClient(api.app)
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json={"image_base64": "a", "extra": 1}).status_code == 422


def test_real_model_inference():
    from pathlib import Path

    from plantvision.inference import predict

    root = Path(__file__).resolve().parents[1]
    if not (root / "models/selected.pt").exists():
        pytest.skip("Actual trained artifact remains local")
    image = next((root / "data/raw/test").rglob("*.jpg"))
    p = predict(image.read_bytes(), root / "models/selected.pt")
    assert len(p["probabilities"]) == 3
    assert abs(sum(p["probabilities"].values()) - 1) < 1e-6
    client = TestClient(api.app)
    response = client.post(
        "/predict", json={"image_base64": base64.b64encode(image.read_bytes()).decode()}
    )
    assert response.status_code == 200
    assert response.json()["label"] == p["label"]
    assert client.post("/predict", json={"image_base64": "invalid!"}).status_code == 422
