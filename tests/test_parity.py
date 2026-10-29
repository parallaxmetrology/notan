"""Parity with the Image Structure Reader this package was extracted from.

Skipped unless the `isr` package can be imported (set ISR_PATH to its folder).

One intentional difference: on neutral (colourless) images the ISR stretches chroma
rounding noise to full scale; this package treats it as no colour. So the relative Notan
mass matches exactly on coloured images and differs, by design, on neutral ones. The
construction signature reads raw values and matches everywhere.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("ISR_PATH"):
    sys.path.insert(0, os.environ["ISR_PATH"])
isr_notan = pytest.importorskip("isr.notan")
isr_construction = pytest.importorskip("isr.construction")
isr_fields = pytest.importorskip("isr.fields")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import blend_mass  # noqa: E402

from parallax_notan import construction_read, notan_mass, notan_readout  # noqa: E402
from parallax_notan.colour import load_rgb  # noqa: E402
from parallax_notan.stimuli import build  # noqa: E402

S = build()
REAL = Path(__file__).resolve().parent / "data" / "counterweight.png"
S["counterweight"] = load_rgb(REAL)
NAMES = sorted(S)
COLOURED = ["colour_built", "counterweight"]
NEUTRAL = [n for n in NAMES if n not in COLOURED and n != "flat"]   # flat has no chroma range at all


@pytest.mark.parametrize("name", NAMES)
def test_absolute_mass_identical(name):
    assert np.array_equal(notan_mass(S[name], True), isr_notan.notan_mass(S[name], True))


@pytest.mark.parametrize("name", COLOURED + ["flat"])
def test_relative_mass_identical_on_coloured_images(name):
    assert np.array_equal(notan_mass(S[name]), isr_notan.notan_mass(S[name]))


@pytest.mark.parametrize("name", NEUTRAL)
def test_relative_mass_differs_on_neutral_images(name):
    assert not np.array_equal(notan_mass(S[name]), isr_notan.notan_mass(S[name]))


@pytest.mark.parametrize("name", COLOURED)
def test_readout_identical_on_coloured_images(name):
    ours = notan_readout(S[name])
    for k, v in isr_notan.notan_readout(S[name]).items():
        if k != "note":
            assert ours[k] == v, k


@pytest.mark.parametrize("name", NAMES)
def test_construction_identical(name):
    ours = construction_read(S[name])
    for k, v in isr_construction.construction_read(S[name]).items():
        if k != "note":
            assert ours[k] == v, k


@pytest.mark.parametrize("name", NAMES)
def test_baseline_is_the_isr_blend(name):
    assert np.array_equal(blend_mass(S[name]), isr_fields.evidence_fields(S[name])["mass"])
