<div align="center">

# COVID-19 Chest X-Ray Classification

### A Comparative Study of Dense, CNN, Deep CNN and LSTM Architectures

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-metrics-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

**3-class classification of chest X-rays: `Covid` · `Normal` · `Viral Pneumonia`**

</div>

---

## Overview

This project investigates whether a neural network can automatically distinguish
**COVID-19**, **Viral Pneumonia** and **Normal** chest X-rays.

The dataset is small (251 training images), which makes it a good test of how
architecture choices behave when data is limited. Instead of tuning one model, four
architectures of increasing complexity are trained under **identical conditions**
(same optimizer, learning rate, batch size and stopping rule) so that differences in
results come from the architecture itself.

## Key Result

> A compact **3-block CNN** is the best model, reaching **90.91% test accuracy**
> with balanced performance across all three classes, while using **17× fewer
> parameters** than the Dense baseline. Adding more depth and regularization made
> results much worse.

| Model | Parameters | Test accuracy | Macro F1 | Errors (of 66) |
|---|---|---|---|---|
| Dense baseline | 20.55 M | 84.85% | 0.84 | 10 |
| **CNN** | **1.21 M** | **90.91%** | **0.90** | **6** |
| Deep CNN (Regularized) | 1.57 M | 60.61% | 0.47 | 26 |
| Deep RNN (LSTM) | 0.35 M | 69.70% | 0.68 | 20 |

**Per-class recall** (share of true cases correctly identified):

| Model | Covid | Normal | Viral Pneumonia |
|---|---|---|---|
| Dense baseline | 0.88 | 0.75 | 0.90 |
| **CNN** | **0.96** | **0.95** | 0.80 |
| Deep CNN (Regularized) | 0.96 | 0.00 | 0.75 |
| Deep RNN (LSTM) | 0.88 | 0.50 | 0.65 |

## Main Findings

1. **The simple CNN wins.** It made the fewest errors (6 of 66) and was the only
   model with recall of at least 0.80 on every class.
2. **Architecture mattered more than size or depth.** Ranking by accuracy was
   CNN > Dense > LSTM > Deep CNN. Parameter count did not predict performance.
3. **More depth and regularization hurt.** The Deep CNN fell from 90.91% to 60.61%,
   never predicted the Normal class, and its validation loss climbed from 1.07 to
   2.49 while training loss stayed low.
4. **Spatial structure is what the task needs.** The Dense model ignores it, the
   LSTM only partly keeps it, and the CNN exploits it directly, and results follow
   that order.
5. **Covid is the easiest class.** The hard boundary is **Normal vs Viral
   Pneumonia**, where most remaining errors occur.

## Models Compared

| # | Model | Idea | Architecture |
|---|---|---|---|
| 1 | **Dense baseline** | Treat every pixel independently | `Flatten` → `BatchNorm` → `Dense(300)` → `BatchNorm` → `Dense(100)` → `Dense(3)` |
| 2 | **CNN** | Detect local patterns (edges, textures) | 3 × `Conv2D` + `MaxPool` (16→32→64) → `Dropout(0.5)` → `Dense(64)` → `Dense(3)` |
| 3 | **Deep CNN (Regularized)** | Test whether more depth and regularization help | 4 × `Conv2D` + `BatchNorm` + `MaxPool` (32→64→128→128) → `Dropout` → `Dense(128)` → `Dropout` → `Dense(3)` |
| 4 | **Deep RNN (LSTM)** | Read image rows as a sequence | `Reshape(150, 450)` → `LSTM(128)` → `LSTM(64)` → `Dense(128)` → `Dense(3)` |

**Common training setup (identical for all four):**

| Setting | Value |
|---|---|
| Image size | 150 × 150 RGB, pixels scaled to [0, 1] |
| Batch size | 16 |
| Loss | Categorical cross-entropy |
| Optimizer | Adam, learning rate 0.0001 |
| Max epochs | 30 |
| Early stopping | Monitor `val_accuracy`, patience 5, restore best weights |

## Dataset

| Split | Images | Covid | Normal | Viral Pneumonia |
|---|---|---|---|---|
| Train | 251 | – | – | – |
| Test | 66 | 26 | 20 | 20 |

Images are read straight from class-named folders, so the expected layout is:

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

> The dataset is **not included** in this repository. Download it separately and
> place it as shown above. <!-- TODO: add dataset source link and licence -->

## Repository Structure

```
.
├── COVID-19-Chest-X-Ray-Classification.ipynb   # Full analysis: pipeline, 4 models, conclusions
├── Asset/                                       # Banner image used in this README
├── Covid19-dataset/                             # Dataset (see Provenance & License above)
│   ├── train/
│   │   ├── Covid/
│   │   ├── Normal/
│   │   └── Viral Pneumonia/
│   └── test/
│       ├── Covid/
│       ├── Normal/
│       └── Viral Pneumonia/
├── Training_history_jason/                      # Saved training curves (created when you run the notebook)
│   ├── dense_history.json
│   ├── cnn_history.json
│   ├── deep_cnn_history.json
│   └── deep_rnn_history.json
├── anaconda_projects/                           # Local Anaconda project files (git-ignored)
└── README.md
```

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/sanjibSamadder/<your-repo-name>.git
cd <your-repo-name>
```

**2. Create an environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install tensorflow numpy scikit-learn jupyter
```

The notebook was developed with **Python 3.10** on an Apple M2 (TensorFlow with the
Metal plugin), but it runs on CPU or any TensorFlow-supported GPU.

**3. Point the notebook at your data**

In the data pipeline cell, update the two paths to where you saved the dataset:

```python
train_dir = "/path/to/Covid19-dataset/train"
test_dir  = "/path/to/Covid19-dataset/test"
```

**4. Run the notebook**

```bash
jupyter notebook COVID-19-Chest-X-Ray-Classification.ipynb
```

Run the cells from top to bottom. The data pipeline cell must run first, because all
four model cells depend on it.

## Limitations

These results are an initial comparison, not a validated clinical finding.

- **The test set doubles as the validation set.** Early stopping and best-weight
  restoration used the same 66 images later reported as test accuracy, so the scores
  are **optimistically biased**.
- **The test set is very small.** One image is worth about 1.5 percentage points, so
  the 4-image gap between the CNN and Dense model could be chance.
- **Each model was trained once.** Run-to-run variance is unknown, and the Deep CNN's
  collapse may partly reflect a single unlucky initialization.
- **No data augmentation and no hyperparameter tuning** were used.
- **Explanations are hypotheses.** Reasons given for the Deep CNN and LSTM
  underperforming fit the training curves but were not tested with ablations.
- **Not for clinical use.** Nothing here supports diagnostic decisions.

## Provenance & License

**Source:** University of Montreal. Dataset uploaded to Kaggle by Pranav Raikote — [Covid-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset).

**Collection methodology:** The COVID-19 and Normal images were collected from a publicly released GitHub account maintained by University of Montreal professors. The Viral Pneumonia images were sourced from the RSNA (Radiological Society of North America) website.

**License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — you are free to share and adapt the dataset, provided appropriate credit is given and any derivative works are distributed under the same license.

**Citations:**
- Pranav Raikote, *Covid-19 Image Dataset*, Kaggle. https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

## Future Work

- [ ] Create a proper train / validation / test split
- [ ] Add k-fold cross-validation and repeated runs with different seeds
- [ ] Add data augmentation and re-test the Deep CNN
- [ ] Try transfer learning (e.g. ResNet, EfficientNet)
- [ ] Run ablations on the Deep CNN to find what caused the collapse
- [ ] Target the Normal / Viral Pneumonia boundary (class weights, Grad-CAM)

## Author

**Sanjib Samadder**
Data Analyst · MSc Data Science, University of Leicester

[GitHub](https://github.com/sanjibSamadder) · [Portfolio](https://sanjibsamadder.github.io) · [LinkedIn](https://www.linkedin.com/)

## Disclaimer

This project is for educational and portfolio purposes only. It is **not a medical
device** and must not be used for diagnosis or clinical decision-making.
