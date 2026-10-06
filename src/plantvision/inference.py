"""Local state-dict inference without arbitrary pickled object loading."""

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from plantvision.data import CLASSES
from plantvision.images import canonical_rgb
from plantvision.models import build, transform
from plantvision.train import probabilities


def decode_image(content: bytes):
    if not content or len(content) > 3_000_000:
        raise ValueError("Image must be nonempty and at most 3 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width > 4096 or image.height > 4096 or image.width < 16 or image.height < 16:
                raise ValueError("Unsupported image dimensions")
            return canonical_rgb(image)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid image") from exc


def predict(content: bytes, artifact: Path):
    image = decode_image(content)
    state = torch.load(artifact, map_location="cpu", weights_only=True)
    model = build(state["kind"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        z = model(transform(state["image_size"])(image).unsqueeze(0)).numpy()
    p = probabilities(z, state["temperature"])[0]
    return {
        "label": CLASSES[int(np.argmax(p))],
        "probabilities": dict(zip(CLASSES, map(float, p), strict=True)),
        "scope": "Three bean classes only; no out-of-domain or agronomic diagnosis validation",
    }
