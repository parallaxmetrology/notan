"""The comparison: an edge/tone/chroma blend mass, the common way to build a "mass" map.

Sobel edges, local tone deviation and chroma, each percentile-normalised, blended
0.42 / 0.32 / 0.26 and lightly smoothed. Gradient-led blends like this one break a
textured shape into fragments. The Notan gate shows how much less Notan is fooled.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters

from parallax_notan.colour import ensure_rgb01, luminance, rgb_to_lms


def _robust01(x, lo=2.0, hi=98.0):
    x = np.asarray(x, dtype=np.float32)
    a, b = np.percentile(x, [lo, hi])
    if b <= a + 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - a) / (b - a), 0, 1).astype(np.float32)


def blend_mass(image) -> np.ndarray:
    rgb = ensure_rgb01(image)
    lum = luminance(rgb)
    L, M, S = rgb_to_lms(rgb)
    l_, m_, s_ = (np.cbrt(np.clip(c, 0, None)) for c in (L, M, S))
    a = (1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_).astype(np.float32)
    b = (0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_).astype(np.float32)
    chroma = _robust01(np.sqrt(a ** 2 + b ** 2))
    edge = _robust01(filters.sobel(lum))
    tone = _robust01(np.abs(lum - filters.gaussian(lum, sigma=12, preserve_range=True)))
    mass = _robust01(0.42 * edge + 0.32 * tone + 0.26 * chroma)
    return _robust01(ndi.gaussian_filter(mass, sigma=1.2))
