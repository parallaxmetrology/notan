"""Check the browser port (docs/notan.js) against the Python package.

Runs both on the nine images in parallax_notan.stimuli and the four paintings in
docs/paintings/, then compares every output the demo shows:

  input          bytes to float32 (the page's fromRGBA vs load_rgb)    must be equal
  small          the 512 px working image (resize)                     max abs difference
  mass           notan_mass on the working image                       max abs difference
  lost           dependence_map("grayscale")["lost"]                   max abs difference
  v2 / v3 / v5   value_map                                             share of pixels that differ
  readout        notan_readout, rounded as Python rounds               must be equal
  construction   construction_read survivals and reading               must be equal
  round()        exact ties such as 0.03125 (Python rounds to even)    must be equal

The paintings are decoded once by Pillow and handed to the JavaScript as the same RGBA
bytes the page gets from a canvas, so this measures the port, not JPEG decoders. The
stimuli go in as float32 arrays, as the Python tests use them. Needs Node 18 or later.

    python tools/js_parity.py            # prints the table, exits 1 on a failure
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from parallax_notan import construction_read, dependence_map, notan_readout
from parallax_notan.colour import load_rgb
from parallax_notan.notan import notan_mass, resize_work, value_map
from parallax_notan.stimuli import build

ROOT = Path(__file__).resolve().parents[1]

# Tolerances. The port follows numpy/scipy/scikit-image arithmetic, so what is left is
# the last bit of library functions: JavaScript's Math.exp, pow and cbrt are not the C
# library's, and differ in the final binary digit for some inputs. Observed: mass about
# 1e-15. One Gauguin pixel of the working image is one float32 step off because Math.exp
# and the C library disagree in the last bit on one blur weight (and numpy's own result
# can vary across platforms the same way). A real porting error shows up at 1e-3 or more.
TOL_SMALL = 6e-8         # max abs difference in the 512 px image: one float32 step
TOL_MASS = 1e-9          # max abs difference in mass and lost (mass is about 0..1)
TOL_VALUES = 1e-4        # share of pixels allowed to differ in a value map (ties at a threshold)


# Python's round() breaks exact ties to even; readout shares like k/32 hit them.
ROUND_CASES = [(k / 32, 4) for k in range(33)] + [(-k / 32, 4) for k in range(1, 5)] + [
    (0.12345, 4), (0.99995, 4), (5e-05, 4), (1 / 3, 4), (2.5e-05, 4), (0.0625, 3), (0.375, 2)]


def cases() -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """name -> (float32 RGB as the Python package reads it, RGBA bytes for the JS or None)."""
    out = {f"stimulus/{k}": (v, None) for k, v in build().items()}
    for p in sorted((ROOT / "docs" / "paintings").glob("*.jpg")):
        rgb = load_rgb(p)
        rgba = np.concatenate([np.asarray(Image.open(p).convert("RGB")), np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)
        out[f"painting/{p.stem}"] = (rgb, rgba)
    return out


def write_reference(tmp: Path) -> list[dict]:
    meta = []
    for i, (name, (rgb, rgba)) in enumerate(cases().items()):
        rgb = np.ascontiguousarray(rgb, dtype=np.float32)
        h, w = rgb.shape[:2]
        rgb.tofile(tmp / f"{i}.in.f32")
        if rgba is not None:
            np.ascontiguousarray(rgba).tofile(tmp / f"{i}.in.u8")
        small = resize_work(rgb)
        mass = notan_mass(small)
        np.ascontiguousarray(small, dtype=np.float32).tofile(tmp / f"{i}.small.f32")
        mass.astype(np.float64).tofile(tmp / f"{i}.mass.f64")
        for n in (2, 3, 5):
            value_map(small, n, mass).astype(np.float64).tofile(tmp / f"{i}.v{n}.f64")
        dm = dependence_map(rgb)
        dm["lost"].astype(np.float64).tofile(tmp / f"{i}.lost.f64")
        meta.append({
            "i": i, "name": name, "w": w, "h": h, "rgba": rgba is not None,
            "sw": small.shape[1], "sh": small.shape[0],
            "readout": notan_readout(rgb),
            "construction": construction_read(rgb),
            "dep_survival": dm["survival"],
        })
    (tmp / "meta.json").write_text(json.dumps(meta))
    (tmp / "round.json").write_text(json.dumps([[v, d, round(v, d)] for v, d in ROUND_CASES]))
    return meta


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write_reference(tmp)
        try:
            res = subprocess.run(["node", str(ROOT / "tools" / "js_parity.mjs"), str(tmp)],
                                 capture_output=True, text=True)
        except FileNotFoundError:
            print("Node is not installed (needs Node 18 or later): https://nodejs.org", file=sys.stderr)
            return 2
        if res.returncode != 0:
            print(res.stdout, res.stderr, file=sys.stderr)
            return 2
        out = json.loads(res.stdout)
        rows, round_bad = out["rows"], out["round_mismatch"]

    fails = 0
    print(f"{'image':28} {'input':>9} {'small':>9} {'mass':>9} {'lost':>9} {'v2':>7} {'v3':>7} {'v5':>7}  readout  constr")
    for r in rows:
        ok = (r["input"] == 0 and r["small"] <= TOL_SMALL and r["mass"] <= TOL_MASS and r["lost"] <= TOL_MASS
              and all(r[f"v{n}"] <= TOL_VALUES for n in (2, 3, 5))
              and r["readout_equal"] and r["construction_equal"])
        fails += not ok
        print(f"{r['name']:28} {r['input']:9.1e} {r['small']:9.1e} {r['mass']:9.1e} {r['lost']:9.1e} "
              f"{r['v2']:7.1e} {r['v3']:7.1e} {r['v5']:7.1e}  "
              f"{'equal' if r['readout_equal'] else 'DIFF':7}  {'equal' if r['construction_equal'] else 'DIFF':6}"
              f"{'' if ok else '   FAIL ' + json.dumps(r.get('detail', {}))}")
    print("\ninput: bytes to float32 for the paintings (the page's path), must be 0")
    print(f"small: max abs difference in the 512 px working image (tolerance {TOL_SMALL:g}, one float32 step)")
    print(f"mass/lost: max abs difference (tolerance {TOL_MASS:g})")
    print(f"v2/v3/v5: share of pixels that differ (tolerance {TOL_VALUES:g})")
    print(f"{len(rows) - fails}/{len(rows)} images within tolerance")
    n_round = len(ROUND_CASES)
    print(f"round(): {n_round - len(round_bad)}/{n_round} cases match Python (ties to even)"
          + (f"   FAIL {round_bad}" if round_bad else ""))
    return 1 if fails or round_bad else 0


if __name__ == "__main__":
    sys.exit(main())
