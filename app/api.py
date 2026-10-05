"""Strict local image inference service."""

import base64
import binascii
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from plantvision.inference import predict

app = FastAPI(
    title="Bean leaf classifier", description="Research demo: only three labeled bean classes"
)
ARTIFACT = Path(__file__).resolve().parents[1] / "models/selected.pt"


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_base64: str = Field(min_length=1, max_length=4_000_000)


@app.get("/health")
def health():
    if not ARTIFACT.is_file():
        raise HTTPException(503, "Train the local model first")
    return {"status": "ready"}


@app.post("/predict")
def inference(request: Request):
    if not ARTIFACT.is_file():
        raise HTTPException(503, "Train the local model first")
    try:
        content = base64.b64decode(request.image_base64, validate=True)
        return predict(content, ARTIFACT)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(422, str(exc)) from exc
