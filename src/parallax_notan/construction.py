"""Construction signature: is a picture's structure built from value or from colour?

Degrade the image in ways that keep every shape where it is (greyscale, gamma up and
down, histogram flattening), re-read the Notan mass each time, and measure how much of
the original mass survives in place. The degradation that destroys the most is what the
structure depends on.

  built = "chromatic"  greyscale destroys it: the shapes live in colour
  built = "tonal"      it survives greyscale but breaks under a tone-curve change
  built = "mixed"      neither of the above

This is a reading, not a verdict. Deterministic; costs about five Notan reads.
"""
from __future__ import annotations

import numpy as np
from skimage.transform import resize

from .colour import ensure_rgb01, load_rgb, luminance
from .notan import WORK_PX, notan_mass

EPS = 1e-9


def degradations(small: np.ndarray) -> dict[str, np.ndarray]:
    """The battery. Every variant keeps geometry; only appearance changes."""
    out: dict[str, np.ndarray] = {}
    g = luminance(small).astype(np.float64)
    out["grayscale"] = np.repeat(g[:, :, None], 3, axis=2)
    sm = np.clip(small, 0, 1).astype(np.float64)
    out["gamma_up"] = sm ** 1.6
    out["gamma_down"] = sm ** 0.6
    from skimage.exposure import equalize_hist
    lum = g + 1e-6
    ratio = (equalize_hist(lum) / lum)[:, :, None]
    out["contrast_norm"] = np.clip(sm * ratio, 0, 1)
    return out


def survival(ref: np.ndarray, deg: np.ndarray) -> float:
    """Fraction of the reference mass that survives in place: sum(min(ref, deg)) / sum(ref).

    Deliberately sensitive to magnitude. A correlation would call a colour shape that
    collapses under greyscale "unchanged", because it locks onto the tiny leftover pattern.
    """
    tot = float(ref.sum())
    if tot < EPS:
        return 1.0
    return float(np.clip(np.minimum(ref, deg).sum() / tot, 0.0, 1.0))


def _work(image, work_px: int) -> np.ndarray:
    rgb = ensure_rgb01(image) if isinstance(image, np.ndarray) else load_rgb(image)
    H, W = rgb.shape[:2]
    s = work_px / max(H, W)
    return (resize(rgb, (max(1, int(H * s)), max(1, int(W * s))), order=1, anti_aliasing=True)
            if s < 1 else rgb)


def dependence_map(image, degradation: str = "grayscale", work_px: int = WORK_PX) -> dict:
    """Where the structure depends on what a degradation removes.

    Returns the per-pixel mass lost under the degradation (`lost`, same units as the
    mass), the reference mass (`mass`), and the survival score. With the default
    "grayscale", `lost` is where the picture's masses are carried by colour rather than
    value: the whole-image score can say "tonal" while this map shows which parts are
    colour-built.
    """
    small = _work(image, work_px)
    ref = notan_mass(small, absolute=True)
    deg = notan_mass(np.clip(degradations(small)[degradation], 0, 1), absolute=True)
    return {"lost": np.clip(ref - deg, 0, None), "mass": ref, "survival": round(survival(ref, deg), 4)}


def construction_read(image, work_px: int = WORK_PX) -> dict:
    """Survival of the Notan mass under each degradation, and the tonal/chromatic reading."""
    small = _work(image, work_px)
    ref = notan_mass(small, absolute=True)   # raw scale, so a collapse shows as a collapse
    surv = {name: round(survival(ref, notan_mass(np.clip(deg, 0, 1), absolute=True)), 4)
            for name, deg in degradations(small).items()}

    fragile_to = min(surv, key=surv.get)
    gray_surv = surv["grayscale"]
    if gray_surv < 0.5:
        built = "chromatic"
    elif fragile_to in ("gamma_up", "gamma_down"):
        built = "tonal"
    else:
        built = "mixed"
    return {
        "built": built,
        "fragile_to": fragile_to,
        "grayscale_survival": gray_surv,
        "mass_survival": round(float(np.mean(list(surv.values()))), 4),
        "survival_min": round(float(min(surv.values())), 4),
        "per_degradation": surv,
    }
