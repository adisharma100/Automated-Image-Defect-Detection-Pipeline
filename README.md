# Automated Image Defect Detection Pipeline

A computer-vision pipeline that automatically detects and classifies visual defects in product images.
It uses **OpenCV** for preprocessing, a **CNN (PyTorch)** for classification, and **scikit-learn**
for evaluation. The decision threshold is tuned on precision to **minimize false positives** in automated review.

**Stack:** Python, OpenCV, PyTorch, scikit-learn, NumPy, Matplotlib/Seaborn

## Pipeline

```
raw images -> OpenCV preprocessing -> stratified split -> CNN training -> evaluation -> threshold tuning -> inference
              (gray, resize, denoise,   (train/val/test)   (class-weighted,   (confusion matrix,   (precision floor    (CLI)
               CLAHE, normalize)                            early stopping)     P/R/F1)             on val set)
```

## Project structure

```
src/
  config.py             paths and hyperparameters
  generate_synthetic.py synthetic dataset (scratch / crack / dent / good)
  preprocess.py         OpenCV preprocessing + augmentation
  dataset.py            stratified splits + PyTorch Dataset
  model.py              custom CNN and ResNet18 transfer option
  train.py              training loop, class weights, early stopping on val macro-F1
  evaluate.py           confusion matrices, classification report, threshold tuning
  predict.py            single-image / folder inference
```

## Quick start

```bash
pip install -r requirements.txt

python -m src.generate_synthetic --n 300   # or put your own images in data/raw/<class>/
python -m src.train --model cnn            # or --model resnet18
python -m src.evaluate
python -m src.predict data/raw/scratch     # run inference
```

### Using a real dataset
Arrange images as `data/raw/<class_name>/*.png` and edit `CLASSES` in `src/config.py`
(keep a `good` class for defect-free items). Suggested datasets: MVTec AD, NEU Surface Defect (NEU-DET), Severstal steel.

## Methodology

- **Preprocessing (OpenCV):** grayscale, resize, non-local-means denoising, CLAHE contrast enhancement, [0,1] normalization.
- **Augmentation:** flips, 90-degree rotations, brightness/contrast jitter, Gaussian noise (training only).
- **Model:** 4-block Conv-BN-ReLU CNN with global average pooling and dropout; optional pretrained ResNet18.
- **Imbalance handling:** class-weighted cross-entropy; model selection by validation macro-F1.
- **Evaluation:** confusion matrix, per-class precision/recall/F1.
- **False-positive control:** for the binary good-vs-defective decision, the threshold is chosen on the
  validation set to maximize recall subject to `precision >= 0.95`, then reported on the held-out test set.

## Results

Fill these in after running `python -m src.evaluate` (plots are saved in `outputs/`).

| Metric | Default (0.50) | Tuned threshold |
|---|---|---|
| Precision | | |
| Recall | | |
| F1 | | |
| False positives | | |

## Future work
Grad-CAM defect localization, anomaly detection for unseen defect types (autoencoder/PatchCore), ONNX export, REST API for deployment.
