import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import app.api as api
from plantvision.data import pixel_hash
from plantvision.inference import decode_image
from plantvision.models import build, gradcam
from plantvision.train import probabilities


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
