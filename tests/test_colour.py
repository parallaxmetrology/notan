"""Image loading: high bit-depth greyscale files must not be clipped to white."""
import numpy as np
from PIL import Image

from parallax_notan.colour import load_rgb

RAMP = np.linspace(0.0, 1.0, 256)


def _save(tmp_path, arr, name):
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return p


def test_8bit_unchanged(tmp_path):
    a = np.tile((RAMP * 255).round().astype(np.uint8), (4, 1))
    rgb = load_rgb(_save(tmp_path, a, "g8.png"))
    assert rgb.dtype == np.float32 and rgb.shape == (4, 256, 3)
    np.testing.assert_array_equal(rgb[..., 0], a.astype(np.float32) / 255.0)


def test_16bit_png_and_tiff_read_full_range(tmp_path):
    a = np.tile((RAMP * 65535).round().astype(np.uint16), (4, 1))
    for name in ("g16.png", "g16.tif"):
        rgb = load_rgb(_save(tmp_path, a, name))
        assert rgb.dtype == np.float32 and rgb.shape == (4, 256, 3)
        assert abs(float(rgb.mean()) - 0.5) < 1e-3          # was about 0.996 (clipped to white)
        np.testing.assert_allclose(rgb[0, :, 0], RAMP, atol=1e-6)


def test_float_tiff(tmp_path):
    a = np.tile(RAMP.astype(np.float32), (4, 1))
    rgb = load_rgb(_save(tmp_path, a, "f.tif"))
    np.testing.assert_allclose(rgb[0, :, 1], RAMP, atol=1e-6)  # was 0 for everything below 1
