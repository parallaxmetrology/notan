"""Notan mass: the big light and dark shapes of a picture, read the way a painter squints.

Luminance (plus half-weighted OKLab chroma) is blurred at three scales and measured as
distance from the image's own median tone. Texture disappears in the blur, so a brushy
or noisy shape survives as one mass, and a flat ground reads as empty.

Why the median: the dominant tone is treated as the ground. Shapes lighter OR darker than
it are mass. On a picture split evenly between two tones (a black square on white) the
median falls between them, so both read as mass and `field_void` is about 0. That is the
intended reading ("two masses, no empty field"), not an error.

Deterministic. No learned model, no recognition.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from .colour import ensure_rgb01, load_rgb, luminance, oklab_chroma

WORK_PX = 512                     # longest side for reading; the blur scales are in pixels at this size
NOTAN_SCALES = (4.0, 8.0, 16.0)   # the squint ladder
CHROMA_WEIGHT = 0.5               # weight of colour mass relative to value mass
TAU_VOID = 0.06                   # mass below this counts as flat / empty
CHROMA_DUST = 1e-6                # raw chroma below this is rounding noise, not colour
EPS = 1e-9


def _norm01(f: np.ndarray) -> np.ndarray:
    lo, hi = float(f.min()), float(f.max())
    return (f - lo) / (hi - lo) if hi - lo > EPS else np.zeros_like(f, dtype=np.float64)


def chroma_support(rgb: np.ndarray) -> float:
    """98th-percentile OKLab chroma before any rescaling.

    Reported so a reader can see how much colour the image really has. Muted paintings
    still read around 0.01 to 0.04; a neutral grey image reads about 1e-8.
    """
    return float(np.percentile(oklab_chroma(ensure_rgb01(rgb)), 98))


def notan_mass(rgb: np.ndarray, absolute: bool = False) -> np.ndarray:
    """Per-pixel Notan mass for an RGB image in [0, 1].

    absolute=False (default) stretches luminance and chroma to [0, 1] per image, so the
    reading does not depend on overall contrast. absolute=True keeps raw scales, so a
    shape that collapses under a change (see construction.py) reads near zero instead of
    being stretched back up.
    """
    rgb = ensure_rgb01(rgb)
    lum = luminance(rgb).astype(np.float64)
    chroma = oklab_chroma(rgb)
    if absolute:
        L = np.clip(lum, 0.0, 1.0)
        C = np.clip(chroma, 0.0, 1.0)
    else:
        L = _norm01(lum)
        # A neutral image has chroma of about 1e-8 (rounding). Stretching that to [0, 1]
        # would invent full-strength colour, so below the dust level there is no colour.
        C = _norm01(chroma) if float(chroma.max()) >= CHROMA_DUST else np.zeros_like(chroma)
    Lb, Cb = float(np.median(L)), float(np.median(C))
    mass = np.zeros(L.shape, dtype=np.float64)
    for s in NOTAN_SCALES:
        mass += np.abs(gaussian_filter(L, s) - Lb) + CHROMA_WEIGHT * np.abs(gaussian_filter(C, s) - Cb)
    return mass / len(NOTAN_SCALES)


def field_void(mass: np.ndarray, tau: float = TAU_VOID) -> float:
    """Fraction of the frame that is flat or mid-tone (mass below tau).

    Read it as "flat fraction", not "background fraction": both polarities count as mass.
    """
    return float((mass < tau).mean())


def field_dispersion(mass: np.ndarray) -> float:
    """Mass-weighted spread around the mass centroid, divided by the image diagonal."""
    h, w = mass.shape
    tot = float(mass.sum())
    if tot < EPS:
        return 0.0
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float((xx * mass).sum() / tot)
    cy = float((yy * mass).sum() / tot)
    var = float((mass * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / tot)
    return float(np.sqrt(var) / np.sqrt(h * h + w * w))


def resize_work(rgb: np.ndarray, work_px: int = WORK_PX) -> np.ndarray:
    """Downscale so the longest side is work_px (never upscales)."""
    H, W = rgb.shape[:2]
    if max(H, W) <= work_px:
        return rgb
    s = work_px / max(H, W)
    return resize(rgb, (max(1, int(H * s)), max(1, int(W * s))), order=1, anti_aliasing=True)


def value_map(rgb: np.ndarray, n: int = 3, mass: np.ndarray | None = None) -> np.ndarray:
    """A 2, 3 or 5 value Notan image in [0, 1], relative to the median ground.

    2: darker or lighter than the ground.
    3: adds the flat ground (no mass) as mid grey.
    5: splits the darks and the lights each by their own lightness.
    """
    if n not in (2, 3, 5):
        raise ValueError("n must be 2, 3 or 5")
    rgb = ensure_rgb01(rgb)
    if mass is None:
        mass = notan_mass(rgb)
    lum = luminance(rgb).astype(np.float64)
    sq = gaussian_filter(lum, 8.0)
    darker = sq < float(np.median(lum))
    if n == 2:
        return np.where(darker, 0.0, 1.0)
    has = mass >= TAU_VOID
    img = np.full(mass.shape, 0.5)
    img[has & darker] = 0.0
    img[has & ~darker] = 1.0
    if n == 5:
        for side, extreme, near, deeper in ((darker, 0.0, 0.25, np.less),
                                            (~darker, 1.0, 0.75, np.greater)):
            sel = has & side
            if sel.any():
                cut = float(np.median(sq[sel]))
                img[sel] = near
                img[sel & deeper(sq, cut)] = extreme
    return img


def notan_readout(image, work_px: int = WORK_PX, return_mass: bool = False) -> dict:
    """Read one image (path or array) and summarise its Notan mass.

    Returns:
      field_void        fraction of the frame that is flat
      field_dispersion  how spread out the mass is (0 = one point, larger = scattered)
      mass_centroid     [x, y] of the mass, -0.5..0.5 with 0 at the centre
      offcenter_force   distance of that centroid from the centre
      mass_fraction     fraction of the frame that carries mass
      chroma_support    raw chroma level; near 0 means the colour channel is noise
      work_px           size the image was read at
      neutral           True when the image has no colour at all; the mass is then value only
    """
    rgb = ensure_rgb01(image) if isinstance(image, np.ndarray) else load_rgb(image)
    small = resize_work(rgb, work_px)
    mass = notan_mass(small)
    h, w = mass.shape
    tot = float(mass.sum()) + EPS
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float((xx * mass).sum() / tot) / max(w - 1, 1) - 0.5
    cy = float((yy * mass).sum() / tot) / max(h - 1, 1) - 0.5
    out = {
        "field_void": round(field_void(mass), 4),
        "field_dispersion": round(field_dispersion(mass), 4),
        "mass_centroid": [round(cx, 4), round(cy, 4)],
        "offcenter_force": round(float(np.hypot(cx, cy)), 4),
        "mass_fraction": round(float((mass >= TAU_VOID).mean()), 4),
        "chroma_support": float(f"{chroma_support(small):.3g}"),
        "work_px": int(max(small.shape[0], small.shape[1])),
        "neutral": bool(float(oklab_chroma(small).max()) < CHROMA_DUST),
    }
    if return_mass:
        out["mass"] = mass
    return out
