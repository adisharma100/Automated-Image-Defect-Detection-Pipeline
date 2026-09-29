"""OpenCV preprocessing + lightweight augmentation."""
import cv2
import numpy as np
from .config import IMG_SIZE


def preprocess(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """BGR uint8 image -> float32 (1, H, W) tensor-ready array in [0, 1].

    Steps: grayscale -> resize -> non-local-means denoise -> CLAHE -> normalize.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return (gray.astype(np.float32) / 255.0)[None, ...]


def load_and_preprocess(path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return preprocess(img)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flips, 90-degree rotations, brightness/contrast jitter, noise. x is (1,H,W)."""
    img = x[0]
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    if rng.random() < 0.5:
        img = cv2.flip(img, 0)
    img = np.rot90(img, k=int(rng.integers(0, 4)))
    alpha = rng.uniform(0.85, 1.15)   # contrast
    beta = rng.uniform(-0.05, 0.05)   # brightness
    img = img * alpha + beta
    if rng.random() < 0.3:
        img = img + rng.normal(0, 0.02, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)[None, ...].copy()
