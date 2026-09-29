"""Test-set evaluation: confusion matrix, precision/recall/F1, and false-positive-aware thresholding."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, precision_score, recall_score, f1_score)

from .config import *
from .dataset import DefectDataset, load_splits
from .model import get_model


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval(); probs, gts = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy()); gts += y.tolist()
    return np.concatenate(probs), np.array(gts)


def plot_cm(cm, labels, title, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODEL_DIR / "best.pt", map_location=device)
    model = get_model(ckpt["arch"]).to(device); model.load_state_dict(ckpt["state"])

    s = load_splits()
    val_probs, val_y = get_probs(model, DataLoader(DefectDataset(*s["val"]), BATCH_SIZE), device)
    test_probs, test_y = get_probs(model, DataLoader(DefectDataset(*s["test"]), BATCH_SIZE), device)
    OUT_DIR.mkdir(exist_ok=True)

    # 1) Multi-class report
    pred = test_probs.argmax(1)
    print("=== Multi-class report (argmax) ===")
    print(classification_report(test_y, pred, target_names=CLASSES, digits=3))
    plot_cm(confusion_matrix(test_y, pred), CLASSES, "Multi-class confusion matrix", OUT_DIR / "cm_multiclass.png")

    # 2) Binary defect-vs-good; tune threshold on VAL to hit target precision (minimize false positives)
    good = CLASSES.index(GOOD_CLASS)
    to_bin = lambda y: (y != good).astype(int)                 # 1 = defective
    score = lambda p: 1.0 - p[:, good]                          # P(defective)

    prec, rec, thr = precision_recall_curve(to_bin(val_y), score(val_probs))
    ok = np.where(prec[:-1] >= TARGET_DEFECT_PRECISION)[0]
    if len(ok):
        t = float(thr[ok[np.argmax(rec[:-1][ok])]])             # best recall subject to precision floor
    else:
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9); t = float(thr[f1s.argmax()])
    print(f"\nChosen defect threshold (tuned on val, precision >= {TARGET_DEFECT_PRECISION}): {t:.3f}")

    yb, sb = to_bin(test_y), score(test_probs)
    for name, th in [("default", 0.5), ("tuned", t)]:
        pb = (sb >= th).astype(int)
        cm = confusion_matrix(yb, pb, labels=[0, 1])
        print(f"[{name} @ {th:.2f}] precision {precision_score(yb, pb, zero_division=0):.3f} | "
              f"recall {recall_score(yb, pb):.3f} | F1 {f1_score(yb, pb):.3f} | "
              f"false positives {cm[0, 1]} | false negatives {cm[1, 0]}")
        plot_cm(cm, ["good", "defective"], f"Binary CM ({name}, thr={th:.2f})", OUT_DIR / f"cm_binary_{name}.png")

    np.save(MODEL_DIR / "threshold.npy", np.array(t))
    print(f"\nPlots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
