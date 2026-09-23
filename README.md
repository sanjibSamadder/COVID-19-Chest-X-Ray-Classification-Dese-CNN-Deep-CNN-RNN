# COVID-19 Chest X-Ray Classification
### A Multi-Architecture Deep Learning Comparison
**Sanjib Samadder** 

![Status](https://img.shields.io/badge/Status-Completed-2ECC71)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-D00000?logo=keras&logoColor=white)
![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-88%25%20(CNN)-2ECC71)
![Model](https://img.shields.io/badge/Model-Dense-1F4E79)
![Model](https://img.shields.io/badge/Model-CNN-16A085)
![Model](https://img.shields.io/badge/Model-Deep%20CNN-8E44AD)
![Model](https://img.shields.io/badge/Model-RNN%20(LSTM)-6A6D6D)
![Task](https://img.shields.io/badge/Task-Image%20Classification-F7931E)



![Banner](Banner_image.jpeg)

A comparison of four neural network architectures — Dense, CNN, Deep CNN, and RNN
(LSTM) — for classifying chest X-rays into **Covid**, **Normal**, or **Viral
Pneumonia**, trained and evaluated identically on the same dataset for a fair,
reproducible benchmark.

---

## Problem Statement

Chest X-ray interpretation is a fast, low-cost screening tool for respiratory illness,
but manual review requires clinical expertise and does not scale easily during periods
of high patient volume. Automated classification offers a way to triage and flag
likely cases for prioritized review — but the choice of model architecture materially
affects both accuracy and the *type* of errors a system makes, which matters
significantly in a diagnostic context.

This project addresses the question: **which neural network architecture is best
suited to classifying chest X-rays into Covid, Normal, or Viral Pneumonia, and why —
demonstrated with a controlled, like-for-like comparison rather than a single model
built in isolation?**

The analysis uses the
[COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
(Kaggle), containing 251 training and 66 test chest X-ray images across the three
classes (test set: Covid 26, Normal 20, Viral Pneumonia 20).

---

## Methodology

### 1. Data Pipeline
Images were loaded directly from class-labeled folders (`train/Covid`,
`train/Normal`, `train/Viral Pneumonia`, and matching `test/` folders) using Keras's
`image_dataset_from_directory`, resized to 150×150, and pixel values rescaled from
0-255 to 0-1. All four models were trained on this identical pipeline to keep the
comparison fair.

### 2. Model Architectures
Four architectures were built and evaluated under identical training conditions
(Adam optimizer, `categorical_crossentropy` loss, `EarlyStopping` with
`patience=5` and `restore_best_weights=True`):

| Model | Architecture | Rationale |
|---|---|---|
| **Dense (baseline)** | `Flatten` -> `Dense(300)` -> `Dense(100)` -> `Dense(3, softmax)` | No convolutional layers -- establishes a baseline with no spatial inductive bias |
| **CNN** | 3x `Conv2D` + `MaxPooling2D` blocks -> `Dropout` -> `Dense` | Exploits local spatial patterns (edges, textures, opacities) native to image data |
| **Deep CNN (regularized)** | 4x `Conv2D` + `BatchNormalization` + `MaxPooling2D`, heavier `Dropout` | Tests whether added depth and stronger regularization improve results on a small dataset |
| **RNN (LSTM)** | Each image row treated as one time-step in a 150-step sequence, fed through stacked `LSTM` layers | Included as a deliberate architectural mismatch -- RNNs are built for sequential data, not static images -- to demonstrate *why* architecture-data fit matters with an actual result, not just an assertion |

### 3. Evaluation
Each model was evaluated on the full 66-image test set using per-class precision,
recall, and F1-score, plus a confusion matrix -- not just a single overall accuracy
number, since aggregate accuracy can hide a model performing well on one class while
failing on another.

---

## Results

| Rank | Model | Accuracy | Macro F1 | Covid Recall | Normal Recall | Viral Pneumonia Recall |
|---|---|---|---|---|---|---|
| 1 | **CNN** | **0.88** | **0.87** | 1.00 | 0.80 | 0.80 |
| 2 | Dense (baseline) | 0.71 | 0.69 | 0.88 | 0.70 | 0.50 |
| 3 | RNN (LSTM) | 0.70 | 0.68 | 0.85 | 0.50 | 0.70 |
| 4 | Deep CNN (regularized) | 0.39 | 0.19 | 1.00 | 0.00 | 0.00 |

> **Deep CNN note:** this model collapsed during training -- it predicted "Covid" for
> every single test image, regardless of true class. The 0.39 accuracy is an artifact
> of Covid's share of the test set (26/66), not genuine skill (Normal and Viral
> Pneumonia both score 0.00 precision/recall). This is most likely caused by
> regularization that was too aggressive (dropout/BatchNorm combined with the added
> depth) for a training set of only 251 images, and is reported here transparently as
> a training issue to debug rather than a fair architectural result.

**CNN was selected as the best model**, outperforming all three alternatives on every
quality metric while producing the most balanced performance across all three classes.

---

## 🏆 Best Model: CNN

- **Highest, most balanced performance** -- leads on both accuracy (0.88) and macro F1
  (0.87), with all three classes scoring F1 >= 0.80.
- **Perfect Covid recall (1.00)** with precision 0.96 -- no missed Covid cases, which
  matters most in a screening context.
- **Errors land in the safer direction** -- remaining mistakes are between Normal and
  Viral Pneumonia, not spilling into missed Covid cases.
- **Stable training** -- validation loss drops quickly and stays flat, with no
  overfitting or oscillation across epochs.
- **Architecturally appropriate** -- convolutional layers naturally capture the local
  spatial patterns (opacities, infiltrates) relevant to X-ray interpretation.

**Why the other models fall short:**

| Model | Limitation |
|---|---|
| Dense (baseline) | No spatial inductive bias -- every pixel treated independently, capping performance; particularly weak on Viral Pneumonia (0.50 recall) |
| RNN (LSTM) | Sequence models are a structural mismatch for static images; the row-by-row sequence trick discards genuine 2D spatial structure, reflected in poor Normal-class recall (0.50) |
| Deep CNN | Currently non-functional (see note above) -- likely fixable with lighter regularization and/or a lower learning rate, but not a valid comparison point as trained |

Full per-model classification reports, confusion matrices, and training curves are in
the notebook.

---

## Repository Structure

```
Covid_XRay_Classification/
├── README.md                             # This file
├── banner.png                             # README banner image
├── covid_xray_model_comparison.ipynb      # Full training, evaluation, and comparison notebook
└── Covid19-dataset/
    ├── train/
    │   ├── Covid/
    │   ├── Normal/
    │   └── Viral Pneumonia/
    └── test/
        ├── Covid/
        ├── Normal/
        └── Viral Pneumonia/
```

---

## Environment & Reproducibility

**Python version:** 3.10

**Key packages:** tensorflow (or tensorflow-macos + tensorflow-metal on Apple
Silicon), scikit-learn, numpy, pandas, matplotlib, seaborn, jupyter

```bash
conda create -n tf_env python=3.10 -y
conda activate tf_env
pip install tensorflow-macos tensorflow-metal scikit-learn numpy jupyter ipykernel matplotlib seaborn pandas
```

(Use `pip install tensorflow` instead of `tensorflow-macos`/`tensorflow-metal` on
non-Apple-Silicon machines.)

**Data:** Download the dataset from
[Kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) and
place it in the repository root, matching the folder structure above, before running
the notebook.

---

## Usage

1. Download and place the dataset as shown in **Repository Structure** above.
2. Update the `train_dir` / `test_dir` paths at the top of the notebook to point to
   your local copy.
3. Run all cells top to bottom.

---

## Next Steps

- Debug and re-run the Deep CNN (lower dropout rate and/or learning rate, verify
  BatchNormalization placement) for a fair 4-way comparison.
- Try transfer learning (e.g. EfficientNetB0 or MobileNetV2 pretrained on ImageNet) as
  a fifth comparison point -- likely to outperform the from-scratch CNN given the small
  training set (251 images).
- Replace the current test-as-validation setup with a proper train/validation/test
  split, keeping the test set untouched until final reporting.
- Given the small test set (66 images), consider k-fold cross-validation for a more
  robust accuracy estimate.

---

## License

**Dataset License:** the source dataset is provided on Kaggle for research and
educational use -- see the
[dataset page](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
for full terms.

This project's own code and analysis (notebook, README, visualizations) are shared for
portfolio and educational purposes. Feel free to reference or build on the methodology
with attribution.

---

**Sanjib Samadder**

**📬 Let's connect!** I'm open to discussions about data science, machine learning, and
collaborative projects.

[![Email](https://img.shields.io/badge/Email-skilled.sanjib%40gmail.com-red?logo=gmail)](mailto:skilled.sanjib@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-sanjibSamadder-181717?logo=github)](https://github.com/sanjibSamadder)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sanjib%20Samadder-0A66C2?logo=linkedin)](https://linkedin.com/in/sanjib-samadder)
