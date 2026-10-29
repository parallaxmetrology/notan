"""Image loading, luminance and OKLab chroma.

These match the Image Structure Reader's own helpers exactly for 8-bit images (see
tests/test_parity.py). `load_rgb` additionally reads 16-bit and float files correctly,
which the ISR's loader does not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as float32 RGB in [0, 1], honouring EXIF orientation.

    16-bit and 32-bit greyscale files (Pillow modes "I;16..." and "I") are scaled from the
    16-bit range, and float files (mode "F") are read as floats. Pillow's convert("RGB")
    clips those modes to 0..255 first, which reads a 16-bit greyscale scan as almost white.
    8-bit files are read exactly as before.
    """
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode.startswith("I;16") or img.mode == "I":
        grey = np.asarray(img, dtype=np.float64) / 65535.0
        return np.repeat(np.clip(grey, 0.0, 1.0).astype(np.float32)[:, :, None], 3, axis=2)
    if img.mode == "F":
        return ensure_rgb01(np.asarray(img, dtype=np.float32))
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def ensure_rgb01(image) -> np.ndarray:
    """Return float32 RGB in [0, 1] from a path, a greyscale array, RGB or RGBA (0-1 or 0-255)."""
    if isinstance(image, (str, Path)):
        return load_rgb(image)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise ValueError(f"Expected RGB/RGBA image, got {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max(initial=0) > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luma on display (gamma-encoded) values, in [0, 1]."""
    rgb = ensure_rgb01(rgb)
    return np.clip(0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2], 0, 1)


def _degamma(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lms(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear sRGB to LMS with the matrix from Ottosson's OKLab paper."""
    r = _degamma(rgb[:, :, 0].astype(np.float64))
    g = _degamma(rgb[:, :, 1].astype(np.float64))
    b = _degamma(rgb[:, :, 2].astype(np.float64))
    L = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    M = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    S = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    return L, M, S


def oklab_chroma(rgb: np.ndarray) -> np.ndarray:
    """OKLab chroma, sqrt(a^2 + b^2), per pixel."""
    L, M, S = rgb_to_lms(ensure_rgb01(rgb))
    l_, m_, s_ = (np.cbrt(np.clip(L, 0, None)), np.cbrt(np.clip(M, 0, None)),
                  np.cbrt(np.clip(S, 0, None)))
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.hypot(a, b).astype(np.float64)
