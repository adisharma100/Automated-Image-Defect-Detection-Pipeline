from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

CLASSES = ["good", "scratch", "crack", "dent"]   # folder names under data/raw
GOOD_CLASS = "good"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
PATIENCE = 6
SEED = 42
# Minimum precision on the "defect" decision, used for threshold tuning
TARGET_DEFECT_PRECISION = 0.95
