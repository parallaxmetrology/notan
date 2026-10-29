# Changelog

## 0.1.0 (2026-10-29)

First public release, extracted from the Image Structure Reader's Notan and construction modules.

- `notan_mass`, `notan_readout`, `value_map` (2, 3 or 5 values), `construction_read`, and the new `dependence_map` (per-pixel mass lost in greyscale).
- `parallax-notan` CLI: `read` and `map`.
- Nine known-answer test images (`parallax_notan.stimuli`) and the gates they were locked to.
- Fix relative to the source: on neutral (colourless) images, chroma rounding noise (about 1e-8) was stretched to full scale and made up a third of the mass. Chroma below 1e-6 now counts as no colour. Readings on coloured images are unchanged; the construction signature is unchanged everywhere. `tests/test_parity.py` checks both.
- Browser demo in `docs/` (for GitHub Pages): a JavaScript port of the mass, value maps, readout, `dependence_map` and `construction_read`, checked against the Python by `tools/js_parity.py`.
- Fix: `load_rgb` read 16-bit greyscale PNG and TIFF files almost entirely white, and float TIFFs almost entirely black, because Pillow's `convert("RGB")` clips those modes to 0..255. They are now scaled from their own range. 8-bit files load exactly as before (`tests/test_colour.py`).
