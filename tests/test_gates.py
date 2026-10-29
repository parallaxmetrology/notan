"""Known-answer gates. The predictions were written before the code was trusted.

Run:  pytest            or   python tests/test_gates.py   (prints the table)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import blend_mass  # noqa: E402

from parallax_notan import construction_read, notan_readout  # noqa: E402
from parallax_notan.notan import field_dispersion, field_void, notan_mass  # noqa: E402
from parallax_notan.stimuli import build  # noqa: E402

S = build()
TAU_BLEND = 0.15


def _void(name):
    return field_void(notan_mass(S[name]))


def _disp(name):
    return field_dispersion(notan_mass(S[name]))


def _blend_void(name):
    return float((blend_mass(S[name]) < TAU_BLEND).mean())


# --- Notan: placement and spread ---------------------------------------------------

def test_flat_is_emptier_than_full():
    assert _void("flat") > _void("full")


def test_centered_is_emptier_than_full():
    assert _void("centered") > _void("full")


def test_scattered_spreads_more_than_centered():
    assert _disp("scattered") > _disp("centered")


def test_flat_has_no_spread():
    assert _disp("flat") < 0.02


# --- Notan: texture ------------------------------------------------------------------

def test_texture_does_not_change_notan_void():
    assert abs(_void("smooth_blob") - _void("textured_blob")) < 0.03


def test_blend_is_more_texture_fooled_than_notan():
    d_notan = abs(_void("smooth_blob") - _void("textured_blob"))
    d_blend = abs(_blend_void("smooth_blob") - _blend_void("textured_blob"))
    assert d_blend > 3 * max(d_notan, 1e-3)


# --- Construction ----------------------------------------------------------------------

def test_colour_built_collapses_in_greyscale():
    r = construction_read(S["colour_built"])
    assert r["grayscale_survival"] < 0.5
    assert r["fragile_to"] == "grayscale"
    assert r["built"] == "chromatic"


def test_tonal_built_survives_greyscale():
    r = construction_read(S["tonal_built"])
    assert r["grayscale_survival"] > 0.7
    assert r["built"] != "chromatic"


def test_dependence_map_finds_the_colour_shape():
    from parallax_notan import dependence_map
    co = dependence_map(S["colour_built"])
    to = dependence_map(S["tonal_built"])
    assert co["lost"].sum() > 0.9 * co["mass"].sum()     # nearly all of it is lost
    assert to["lost"].sum() < 0.01 * to["mass"].sum()    # nearly none of it is lost


# --- Field health ----------------------------------------------------------------------

def _value_only_mass(img):
    from scipy.ndimage import gaussian_filter

    from parallax_notan.colour import luminance
    from parallax_notan.notan import NOTAN_SCALES, _norm01
    L = _norm01(luminance(img).astype(np.float64))
    Lb = float(np.median(L))
    return sum(np.abs(gaussian_filter(L, s) - Lb) for s in NOTAN_SCALES) / len(NOTAN_SCALES)


def test_neutral_grey_has_no_invented_colour():
    # Rounding noise in the chroma channel must not be stretched into "colour".
    assert np.allclose(notan_mass(S["grey_step"]), _value_only_mass(S["grey_step"]))
    assert notan_readout(S["grey_step"])["neutral"] is True


def test_real_colour_still_counts():
    assert notan_readout(S["colour_built"])["neutral"] is False
    assert not np.allclose(notan_mass(S["colour_built"]), _value_only_mass(S["colour_built"]))


if __name__ == "__main__":
    print(f"{'stimulus':14s} {'notan_void':>10s} {'notan_disp':>10s} {'blend_void':>10s}")
    for k in ("flat", "centered", "scattered", "full", "smooth_blob", "textured_blob"):
        print(f"{k:14s} {_void(k):10.3f} {_disp(k):10.3f} {_blend_void(k):10.3f}")
    dn = abs(_void("smooth_blob") - _void("textured_blob"))
    db = abs(_blend_void("smooth_blob") - _blend_void("textured_blob"))
    print(f"\ntexture change in void: Notan {dn:.4f}, blend {db:.4f}  (blend / Notan = {db / max(dn, 1e-9):.0f}x)")
    for k in ("colour_built", "tonal_built"):
        r = construction_read(S[k])
        print(f"{k:13s} built={r['built']:9s} fragile_to={r['fragile_to']:13s} greyscale survival={r['grayscale_survival']:.3f}")
    print("grey_step neutral:", notan_readout(S["grey_step"])["neutral"])
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
