"""Generate a synthetic metal-surface defect dataset so the pipeline runs end to end.

Replace data/raw/<class>/ with real images (e.g. MVTec AD, NEU-DET) for real use.
"""
import argparse
import cv2
import numpy as np
from .config import RAW_DIR, CLASSES


def base_texture(rng, size=200):
    img = rng.normal(140, 12, (size, size)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    grad = np.linspace(-10, 10, size, dtype=np.float32)[None, :]
    img += grad * rng.uniform(-1, 1)
    return img


def add_scratch(img, rng):
    h, w = img.shape
    x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
    ang = rng.uniform(0, np.pi)
    length = int(rng.integers(40, 120))
    x2, y2 = int(x1 + length * np.cos(ang)), int(y1 + length * np.sin(ang))
    cv2.line(img, (x1, y1), (x2, y2), float(rng.uniform(190, 230)), int(rng.integers(1, 3)))


def add_crack(img, rng):
    h, w = img.shape
    x, y = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
    for _ in range(int(rng.integers(8, 20))):
        nx, ny = x + int(rng.integers(-10, 11)), y + int(rng.integers(-10, 11))
        cv2.line(img, (x, y), (nx, ny), float(rng.uniform(30, 70)), 1)
        x, y = int(np.clip(nx, 0, w - 1)), int(np.clip(ny, 0, h - 1))


def add_dent(img, rng):
    h, w = img.shape
    cx, cy = int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40))
    r = int(rng.integers(12, 30))
    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
    shade = np.zeros_like(img)
    shade[mask] = (((xx - cx) + 0 * yy)[mask] / r) * 35   # directional shading = dent look
    img += cv2.GaussianBlur(shade, (0, 0), 3)


def main(n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    makers = {"scratch": add_scratch, "crack": add_crack, "dent": add_dent}
    for cls in CLASSES:
        out = RAW_DIR / cls
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = base_texture(rng)
            if cls in makers:
                makers[cls](img, rng)
            img = np.clip(img, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out / f"{cls}_{i:04d}.png"), img)
    print(f"Generated {n_per_class * len(CLASSES)} images in {RAW_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300, help="images per class")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.n, a.seed)
