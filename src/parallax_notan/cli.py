"""Command line: `parallax-notan read IMAGE...` and `parallax-notan map IMAGE -o OUT.png`."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from PIL import Image

from .colour import load_rgb
from .construction import construction_read
from .notan import notan_mass, notan_readout, resize_work, value_map


def _read(args) -> int:
    results = {}
    for path in args.images:
        r = notan_readout(path)
        if not args.no_construction:
            r["construction"] = construction_read(path)
        results[path] = r
    out = results[args.images[0]] if len(args.images) == 1 else results
    print(json.dumps(out, indent=2))
    return 0


def _map(args) -> int:
    rgb = resize_work(load_rgb(args.image))
    mass = notan_mass(rgb)
    if args.values:
        img = value_map(rgb, args.values, mass)
    else:
        img = 1.0 - np.clip(mass / max(float(mass.max()), 1e-9), 0, 1)   # dark = more mass
    Image.fromarray((img * 255).astype(np.uint8)).save(args.output)
    print(args.output)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="parallax-notan", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="print the Notan readout and construction signature as JSON")
    r.add_argument("images", nargs="+")
    r.add_argument("--no-construction", action="store_true", help="skip the slower construction read")
    r.set_defaults(func=_read)
    m = sub.add_parser("map", help="save the Notan mass as a greyscale PNG (dark = more mass)")
    m.add_argument("image")
    m.add_argument("-o", "--output", required=True)
    m.add_argument("--values", type=int, choices=(2, 3, 5), help="posterise into 2, 3 or 5 values instead")
    m.set_defaults(func=_map)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
