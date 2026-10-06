"""Canonical single-frame pixels shared by audit, training and inference."""

from PIL import Image, ImageOps


def canonical_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF display orientation and detach RGB pixels from the source file."""
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Multiframe images are not supported")
    return ImageOps.exif_transpose(image).convert("RGB")
