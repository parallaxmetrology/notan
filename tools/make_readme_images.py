"""Build the README figures into docs/images/. Run from the repo root:

    python tools/make_readme_images.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from baseline import blend_mass  # noqa: E402

from parallax_notan import construction_read, dependence_map, notan_mass, notan_readout  # noqa: E402
from parallax_notan.colour import load_rgb, luminance, oklab_chroma  # noqa: E402
from parallax_notan.notan import NOTAN_SCALES, _norm01, resize_work, value_map  # noqa: E402
from parallax_notan.stimuli import build  # noqa: E402

OUT = ROOT / "docs" / "images"
PAINTINGS = ROOT / "docs" / "paintings"
BG, INK, SUB = (255, 255, 255), (30, 30, 30), (110, 110, 110)
GAP, TITLE_H = 14, 44
FONT = ImageFont.load_default(size=17)
SMALL = ImageFont.load_default(size=14)

WORKS = [
    ("caravaggio", "Caravaggio, The Calling of Saint Matthew, 1599-1600"),
    ("gauguin", "Gauguin, Vision after the Sermon, 1888"),
    ("bruegel", "After Bruegel, Landscape with the Fall of Icarus, c. 1560s"),
    ("jeong", "Jeong Seon, Clearing after Rain on Mount Inwang, 1751"),
]


def to_img(a: np.ndarray) -> Image.Image:
    a = np.clip(a, 0, 1)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, 2)
    return Image.fromarray((a * 255).round().astype(np.uint8))


def mass_view(mass: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Dark = more mass."""
    s = scale if scale else max(float(np.percentile(mass, 99.5)), 1e-9)
    return 1.0 - np.clip(mass / s, 0, 1)


def fit(im: Image.Image, h: int) -> Image.Image:
    return im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)


def row(tiles: list[tuple[Image.Image, str, str]], h: int, caption: str = "") -> Image.Image:
    """tiles: (image, title, subtitle). All scaled to height h, titles above each tile,
    and an optional caption line above the whole row."""
    ims = [fit(t[0], h) for t in tiles]
    w = sum(i.width for i in ims) + GAP * (len(ims) - 1)
    cap_h = 30 if caption else 0
    title_h = TITLE_H if any(t[1] or t[2] for t in tiles) else 0
    canvas = Image.new("RGB", (w, cap_h + title_h + h), BG)
    d = ImageDraw.Draw(canvas)
    if caption:
        d.text((0, 2), caption, fill=INK, font=ImageFont.load_default(size=19))
    x = 0
    for im, (_, title, sub) in zip(ims, tiles):
        d.text((x, cap_h + 2), title, fill=INK, font=FONT)
        if sub:
            d.text((x, cap_h + 23), sub, fill=SUB, font=SMALL)
        canvas.paste(im, (x, cap_h + title_h))
        x += im.width + GAP
    return canvas


def stack(rows: list[Image.Image], gap: int = 22, pad: int = 16) -> Image.Image:
    w = max(r.width for r in rows) + 2 * pad
    h = sum(r.height for r in rows) + gap * (len(rows) - 1) + 2 * pad
    canvas = Image.new("RGB", (w, h), BG)
    y = pad
    for r in rows:
        canvas.paste(r, (pad, y))
        y += r.height + gap
    return canvas


def save(im: Image.Image, name: str) -> None:
    path = OUT / name
    im.save(path, optimize=True) if name.endswith(".png") else im.save(path, quality=86)
    print(path.relative_to(ROOT), im.size)


def fig_paintings() -> None:
    rows = []
    for key, label in WORKS:
        rgb = resize_work(load_rgb(PAINTINGS / f"{key}.jpg"))
        mass = notan_mass(rgb)
        r = notan_readout(rgb)
        rows.append(row([
            (to_img(rgb), "Painting", ""),
            (to_img(mass_view(mass)), "Notan mass", f"void {r['field_void']:.2f}, dispersion {r['field_dispersion']:.2f}"),
            (to_img(value_map(rgb, 3, mass)), "3 values", "dark / flat ground / light"),
        ], 300, caption=label))
    save(stack(rows), "paintings.jpg")


def fig_texture() -> None:
    S = build()
    rows = []
    for k, label in (("smooth_blob", "Smooth blob"), ("textured_blob", "Same blob, textured")):
        m = notan_mass(S[k])
        b = blend_mass(S[k])
        nv = float((m < 0.06).mean())
        bv = float((b < 0.15).mean())
        rows.append(row([
            (to_img(S[k]), label, ""),
            (to_img(mass_view(m, 0.35)), "Notan mass", f"void {nv:.3f}"),
            (to_img(1.0 - b), "Edge/tone/chroma blend", f"void {bv:.3f}"),
        ], 230))
    save(stack(rows), "texture.png")


def _dependence_overlay(rgb: np.ndarray, lost: np.ndarray, ref: np.ndarray) -> np.ndarray:
    base = 0.40 + 0.55 * luminance(rgb)[..., None] * np.ones(3)    # greyed copy
    a = np.clip(lost / max(float(np.percentile(ref, 99.5)), 1e-9) * 3.0, 0, 0.9)[..., None]
    blue = np.array([0.05, 0.30, 0.95])
    return base * (1 - a) + blue * a


def fig_dependence() -> None:
    rows = []
    for key, label in (WORKS[1], WORKS[0]):
        rgb = resize_work(load_rgb(PAINTINGS / f"{key}.jpg"))
        dm = dependence_map(rgb)
        c = construction_read(rgb)
        rows.append(row([
            (to_img(rgb), "Painting", ""),
            (to_img(_dependence_overlay(rgb, dm["lost"], dm["mass"])),
             "Mass lost in greyscale (blue)",
             f"greyscale keeps {c['grayscale_survival']:.0%} of the mass"),
        ], 330, caption=label))
    save(stack(rows), "colour_dependence.jpg")


def fig_stimuli() -> None:
    S = build()
    order = ["flat", "centered", "scattered", "full", "smooth_blob", "textured_blob",
             "colour_built", "tonal_built", "grey_step"]
    top = row([(to_img(S[k]), k.replace("_", " "), "") for k in order], 120)
    bottom = row([(to_img(mass_view(notan_mass(S[k]), 0.35)), "", "") for k in order], 120)
    save(stack([top, bottom], gap=0), "stimuli.png")


def fig_grey_dust() -> None:
    img = build()["grey_step"]
    L = _norm01(luminance(img).astype(np.float64))
    C = _norm01(oklab_chroma(img))           # the stretch, without the dust guard
    Lb, Cb = float(np.median(L)), float(np.median(C))
    from scipy.ndimage import gaussian_filter
    old = sum(np.abs(gaussian_filter(L, s) - Lb) + 0.5 * np.abs(gaussian_filter(C, s) - Cb)
              for s in NOTAN_SCALES) / len(NOTAN_SCALES)
    new = notan_mass(img)
    scale = float(old.max())
    save(stack([row([
        (to_img(img), "Two greys, no colour", f"raw chroma max {oklab_chroma(img).max():.0e}"),
        (to_img(np.repeat(C[..., None], 3, 2)), "Chroma after stretching", "rounding noise at full scale"),
        (to_img(mass_view(old, scale)), "Mass, stretched", f"mean {old.mean():.2f}"),
        (to_img(mass_view(new, scale)), "Mass, with the fix", f"mean {new.mean():.2f}"),
    ], 200)]), "grey_dust.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_paintings()
    fig_texture()
    fig_dependence()
    fig_stimuli()
    fig_grey_dust()
