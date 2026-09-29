import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from .config import *
from .dataset import DefectDataset, load_splits
from .model import get_model


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(y)
            preds += out.argmax(1).cpu().tolist(); gts += y.cpu().tolist()
    return total / len(loader.dataset), f1_score(gts, preds, average="macro")


def main(args):
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    s = load_splits()
    train_ds = DefectDataset(*s["train"], train=True)
    val_ds = DefectDataset(*s["val"])
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, BATCH_SIZE)

    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=s["train"][1])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    model = get_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    MODEL_DIR.mkdir(exist_ok=True)
    best, bad = -1, 0
    for ep in range(1, args.epochs + 1):
        tl, tf = run_epoch(model, train_dl, device, criterion, opt)
        vl, vf = run_epoch(model, val_dl, device, criterion)
        sched.step(vf)
        print(f"Epoch {ep:02d} | train loss {tl:.4f} F1 {tf:.3f} | val loss {vl:.4f} F1 {vf:.3f}")
        if vf > best:
            best, bad = vf, 0
            torch.save({"state": model.state_dict(), "arch": args.model}, MODEL_DIR / "best.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping."); break
    print(f"Best val macro-F1: {best:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    main(p.parse_args())
