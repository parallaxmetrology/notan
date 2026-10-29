"""Synthetic test images with known answers.

Each stimulus is built so the right reading is not in doubt. They are useful for
scoring any mass, balance or composition measure, not only this package's. Write them
to disk with `python -m parallax_notan.stimuli OUT_DIR`.

The expected relations (what a sound measure should report) are in EXPECTED. They are
the predictions the gates in tests/ were locked to before the code was trusted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

N = 512


def _canvas(v: float = 0.5) -> np.ndarray:
    return np.full((N, N, 3), v, np.float32)


def _disc_mask(cx: int, cy: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:N, :N]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _disc(img: np.ndarray, cx: int, cy: int, r: int, val) -> np.ndarray:
    img[_disc_mask(cx, cy, r)] = val
    return img


def build() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    S: dict[str, np.ndarray] = {}

    # Placement and spread
    S["flat"] = _canvas(0.5)
    S["centered"] = _disc(_canvas(0.6), N // 2, N // 2, 70, 0.2)
    sc = _canvas(0.6)
    for cx, cy in [(110, 110), (400, 130), (256, 256), (120, 400), (400, 400)]:
        _disc(sc, cx, cy, 45, 0.2)
    S["scattered"] = sc
    yy, xx = np.mgrid[0:N, 0:N]
    S["full"] = np.repeat((0.5 + 0.45 * np.sin(xx / 40.0) * np.cos(yy / 40.0))[..., None], 3, 2).astype(np.float32)

    # Texture: the same blob, smooth and noisy. Its mass should not change.
    base = 0.3
    S["smooth_blob"] = _disc(_canvas(0.5), N // 2, N // 2, 90, base)
    tex = _canvas(0.5)
    mask = _disc_mask(N // 2, N // 2, 90)
    noisy = np.clip(base + rng.uniform(-0.18, 0.18, (N, N)), 0, 1)
    for ch in range(3):
        tex[..., ch][mask] = noisy[mask]
    S["textured_blob"] = tex

    # Construction: a shape made only of colour, and one made only of value.
    co = _canvas(0.5)
    m = _disc_mask(N // 2, N // 2, 90)
    for ch, v in enumerate((0.40, 0.55, 0.30)):   # luminance 0.500, same as the ground
        co[..., ch][m] = v
    S["colour_built"] = co
    to = _canvas(0.8)
    to[m] = 0.2
    S["tonal_built"] = to

    # Neutral grey: no colour at all. A measure that finds colour here is reading noise.
    step = _canvas(0.3)
    step[:, N // 2:] = 0.7
    S["grey_step"] = step
    return S


EXPECTED = {
    "flat": "No mass anywhere: dispersion about 0, void highest.",
    "centered": "One central mass: centroid at the centre, low dispersion.",
    "scattered": "Five equal masses: dispersion higher than 'centered'.",
    "full": "Mass everywhere: void lower than 'flat' and 'centered'.",
    "smooth_blob": "One blob. Its pair 'textured_blob' should read the same.",
    "textured_blob": "Same blob with noise inside. Mass and void should match 'smooth_blob'.",
    "colour_built": "Equiluminant green disc: vanishes in greyscale, so colour-built.",
    "tonal_built": "Dark grey disc on light grey: survives greyscale, so not colour-built.",
    "grey_step": "Two greys, zero chroma: any colour reading is noise.",
}


def write(out_dir: str | Path) -> Path:
    from PIL import Image
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, img in build().items():
        Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(out / f"{name}.png")
    (out / "EXPECTED.json").write_text(json.dumps(EXPECTED, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(write(sys.argv[1] if len(sys.argv) > 1 else "stimuli"))
