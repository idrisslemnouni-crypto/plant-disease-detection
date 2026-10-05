"""Pinned download and decoded-pixel duplicate audit."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


def download(root: Path) -> None:
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    for item in json.loads((root / "data/source-manifest.json").read_text()):
        archive = raw / f"{item['split']}.zip"
        if not archive.exists():
            temporary = archive.with_suffix(".part")
            urllib.request.urlretrieve(item["url"], temporary)
            temporary.replace(archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Source archive checksum mismatch")
        if not (raw / item["split"]).exists():
            with zipfile.ZipFile(archive) as zf:
                if any(
                    not (raw / i.filename).resolve().is_relative_to(raw.resolve())
                    for i in zf.infolist()
                ):
                    raise ValueError("Unsafe archive path")
                zf.extractall(raw)


def pixel_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def dhash(image: Image.Image) -> int:
    a = np.asarray(image.convert("L").resize((9, 8)))
    return int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), "big")


def audit(root: Path) -> tuple[pd.DataFrame, dict]:
    rows, seen, removed = [], {}, []
    for split in ["test", "validation", "train"]:
        paths = sorted((root / "data/raw" / split).rglob("*.jpg"))
        if not paths:
            raise ValueError(f"No images for {split}")
        for path in paths:
            label = CLASSES.index(path.parent.name)
            with Image.open(path) as im:
                digest, perceptual = pixel_hash(im), dhash(im)
            relative = path.relative_to(root).as_posix()
            if digest in seen:
                if seen[digest][1] != label:
                    raise ValueError("Identical pixels have conflicting labels")
                removed.append({"path": relative, "kept": seen[digest][0]})
                continue
            seen[digest] = (relative, label)
            rows.append(
                {
                    "path": relative,
                    "split": split,
                    "label": label,
                    "pixel_sha256": digest,
                    "dhash": perceptual,
                }
            )
    table = pd.DataFrame(rows)
    near = []
    # This audit is unsupervised and is not a validated same-plant detector.
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if a["split"] != b["split"] and (a["dhash"] ^ b["dhash"]).bit_count() <= 5:
                near.append(
                    {
                        "a": a["path"],
                        "b": b["path"],
                        "distance": (a["dhash"] ^ b["dhash"]).bit_count(),
                    }
                )
    evidence = {
        "retained_counts": table.groupby("split").size().to_dict(),
        "exact_duplicates_removed": removed,
        "cross_split_dhash_distance_le_5": near,
        "limitation": "Plant, farm and photographer identifiers unavailable; hash screening does not prove independent plants",
    }
    return table.drop(columns="dhash"), evidence
