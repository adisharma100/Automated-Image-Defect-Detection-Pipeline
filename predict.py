"""Inference on an image or folder: python -m src.predict path/to/image_or_dir"""
import sys
from pathlib import Path
import numpy as np
import torch
from .config import MODEL_DIR, CLASSES, GOOD_CLASS
from .preprocess import load_and_preprocess
from .model import get_model


def main(target):
    ckpt = torch.load(MODEL_DIR / "best.pt", map_location="cpu")
    model = get_model(ckpt["arch"]); model.load_state_dict(ckpt["state"]); model.eval()
    thr_file = MODEL_DIR / "threshold.npy"
    thr = float(np.load(thr_file)) if thr_file.exists() else 0.5
    good = CLASSES.index(GOOD_CLASS)

    t = Path(target)
    files = sorted(p for p in (t.glob("*") if t.is_dir() else [t])
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    for f in files:
        x = torch.from_numpy(load_and_preprocess(f))[None]
        with torch.no_grad():
            p = torch.softmax(model(x), 1)[0].numpy()
        p_def = 1 - p[good]
        if p_def < thr:
            verdict = "PASS (good)"
        else:
            masked = p.copy(); masked[good] = -1
            verdict = f"DEFECT: {CLASSES[int(masked.argmax())]}"
        print(f"{f.name}: {verdict}  (P(defect)={p_def:.3f}, threshold={thr:.2f})")


if __name__ == "__main__":
    main(sys.argv[1])
