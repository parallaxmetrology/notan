from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def main():
    out = Path(__file__).parent / "synthetic_counterweight.png"
    img = Image.new("RGB", (640, 420), (232, 229, 216))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse((75, 88, 345, 350), fill=(38, 46, 58, 230), outline=(10, 10, 10, 255), width=4)
    draw.ellipse((360, 50, 470, 160), fill=(120, 62, 45, 220), outline=(45, 20, 10, 220), width=3)
    draw.ellipse((500, 285, 548, 335), fill=(25, 135, 160, 240), outline=(5, 60, 72, 255), width=3)
    draw.line((330, 215, 510, 310), fill=(105, 90, 72, 120), width=8)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    noise = np.random.default_rng(7).normal(0, 3, (420, 640, 3))
    arr = np.clip(np.asarray(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out)
    print(out)


if __name__ == "__main__":
    main()

