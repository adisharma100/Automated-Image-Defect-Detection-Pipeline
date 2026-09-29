import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .config import RAW_DIR, SPLIT_DIR, CLASSES, SEED
from .preprocess import load_and_preprocess, augment


def build_splits(val_size=0.15, test_size=0.15):
    """Stratified train/val/test split saved to data/splits/splits.json."""
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        for p in sorted((RAW_DIR / cls).glob("*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                paths.append(str(p)); labels.append(idx)
    if not paths:
        raise RuntimeError(f"No images found in {RAW_DIR}. Run: python -m src.generate_synthetic")
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=val_size + test_size, stratify=labels, random_state=SEED)
    x_va, x_te, y_va, y_te = train_test_split(
        x_tmp, y_tmp, test_size=test_size / (val_size + test_size), stratify=y_tmp, random_state=SEED)
    splits = {"train": [x_tr, y_tr], "val": [x_va, y_va], "test": [x_te, y_te]}
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "splits.json").write_text(json.dumps(splits))
    return splits


def load_splits():
    f = SPLIT_DIR / "splits.json"
    return json.loads(f.read_text()) if f.exists() else build_splits()


class DefectDataset(Dataset):
    def __init__(self, paths, labels, train=False):
        self.labels = labels
        self.train = train
        self.rng = np.random.default_rng(SEED)
        # Preprocess once and cache in memory (denoising is the slow step)
        self.data = [load_and_preprocess(p) for p in paths]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        if self.train:
            x = augment(x, self.rng)
        return torch.from_numpy(x), torch.tensor(self.labels[i], dtype=torch.long)
