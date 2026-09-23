# COVID-19 Chest X-Ray Classification — Model Comparison

A comparison of four neural network architectures for classifying chest X-rays into **Covid**, **Normal**, or **Viral Pneumonia**, trained and evaluated identically on the same dataset for a fair benchmark.

![Banner](Banner_image.jpeg)

## Dataset

[COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) (Kaggle) — 251 training images, 66 test images, 3 classes (Covid: 26, Normal: 20, Viral Pneumonia: 20 in the test set).

## Models compared

| Model | Architecture |
|---|---|
| **Dense (baseline)** | `Flatten` → `Dense(300)` → `Dense(100)` → `Dense(3, softmax)` — no convolutional layers, treats every pixel independently |
| **CNN** | 3× `Conv2D` + `MaxPooling2D` blocks → `Dropout` → `Dense` — exploits local spatial patterns |
| **Deep CNN (regularized)** | 4× `Conv2D` + `BatchNormalization` + `MaxPooling2D` blocks, heavier `Dropout` — tests whether more depth + stronger regularization helps on a small dataset |
| **RNN (LSTM)** | Each image row treated as one time-step in a 150-step sequence, fed through stacked `LSTM` layers — included for comparison; RNNs are built for sequential data, not static images |

All models were trained with the Adam optimizer, `categorical_crossentropy` loss, and `EarlyStopping` (`monitor="val_accuracy"`, `patience=5`, `restore_best_weights=True`).

## Results

| Rank | Model | Accuracy | Macro F1 | Covid Recall | Normal Recall | Viral Pneumonia Recall |
|---|---|---|---|---|---|---|
| 1 | **CNN** | **0.88** | **0.87** | 1.00 | 0.80 | 0.80 |
| 2 | Dense (baseline) | 0.71 | 0.69 | 0.88 | 0.70 | 0.50 |
| 3 | RNN (LSTM) | 0.70 | 0.68 | 0.85 | 0.50 | 0.70 |
| 4 | Deep CNN (regularized) | 0.39 | 0.19 | 1.00 | 0.00 | 0.00 |

> **Deep CNN note:** this model collapsed during training — it predicted "Covid" for every single test image, regardless of true class. The 0.39 accuracy is an artifact of Covid's share of the test set (26/66), not genuine skill (Normal and Viral Pneumonia both score 0.00 precision/recall). This is most likely caused by regularization that was too aggressive (dropout/BatchNorm combined with the added depth) for a training set of only 251 images, and is flagged here as a training issue to debug, not a fair architectural result.

## 🏆 Best model: CNN

- **Highest, most balanced performance** — leads on both accuracy (0.88) and macro F1 (0.87), with all three classes scoring F1 ≥ 0.80.
- **Perfect Covid recall (1.00)** with precision 0.96 — no missed Covid cases, which matters most in a screening context.
- **Errors land in the safer direction** — remaining mistakes are between Normal and Viral Pneumonia, not spilling into missed Covid cases.
- **Stable training** — validation loss drops quickly and stays flat, with no overfitting or oscillation across epochs.
- **Architecturally appropriate** — convolutional layers naturally capture the local spatial patterns (opacities, infiltrates) relevant to X-ray interpretation.

**Why the others fall short:**
- **Dense:** no spatial inductive bias — every pixel treated independently, capping performance, particularly weak on Viral Pneumonia (0.50 recall).
- **RNN (LSTM):** sequence models are a structural mismatch for static images; the row-by-row sequence trick discards genuine 2D spatial structure, reflected in the poor Normal-class recall (0.50).
- **Deep CNN:** currently non-functional (see note above) — likely fixable with lighter regularization and/or a lower learning rate, but not a valid comparison point as trained.

## Project structure

```
.
├── covid_xray_model_comparison.ipynb   # full training, evaluation, and comparison notebook
└── README.md
```

## Setup

```bash
conda create -n tf_env python=3.10 -y
conda activate tf_env
pip install tensorflow-macos tensorflow-metal scikit-learn numpy jupyter ipykernel matplotlib seaborn pandas
```

(Use `pip install tensorflow` instead of `tensorflow-macos`/`tensorflow-metal` on non-Apple-Silicon machines.)

## Usage

1. Download the dataset from Kaggle and place it so the folder structure looks like:
   ```
   Covid19-dataset/
   ├── train/
   │   ├── Covid/
   │   ├── Normal/
   │   └── Viral Pneumonia/
   └── test/
       ├── Covid/
       ├── Normal/
       └── Viral Pneumonia/
   ```
2. Update the `train_dir` / `test_dir` paths at the top of the notebook to point to your local copy.
3. Run all cells top to bottom.

## Next steps

- Debug and re-run the Deep CNN (lower dropout rate and/or learning rate, verify BatchNormalization placement) for a fair 4-way comparison.
- Try transfer learning (e.g. EfficientNetB0 or MobileNetV2 pretrained on ImageNet) as a fifth comparison point — likely to outperform the from-scratch CNN given the small training set (251 images).
- Replace the current test-as-validation setup with a proper train/validation/test split, keeping the test set untouched until final reporting.
- Given the small test set (66 images), consider k-fold cross-validation for a more robust accuracy estimate.
