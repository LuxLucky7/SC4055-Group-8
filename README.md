# Quantum Kernel SVM vs Classical Baselines for Breast Cancer Classification

**SC4055 Group 8**

Does the way the data is represented (PCA-4, MI-4, MI-decorr-4) or the choice of quantum feature map (entangled angle, ZZ) change how well a **quantum-kernel SVM (QSVM)** classifies breast cancer, compared with a classical **RBF SVM** and **logistic regression** trained on the same data split?

Everything lives in one notebook, `quantum computing.ipynb`, which runs end to end and produces all tables and figures.

## Headline findings

- On the Wisconsin breast cancer data, the best quantum model (QSVM entangled-angle on PCA-4) reaches **0.993 CV ROC-AUC** and **0.991 test ROC-AUC**. The best classical model (RBF SVM on all 30 features) reaches **0.996 CV ROC-AUC** and **0.993 test ROC-AUC**.
- **None of the 18 paired comparisons are significant after Holm correction.** The quantum and classical models are statistically indistinguishable on this dataset.
- Changing the representation moves quantum and classical models by about the same amount (RQ3 interaction tests all have Holm-adjusted p = 1.0).
- Quantum kernels cost far more to build (about 20 to 45 s per experiment for the final kernels, versus a median of 14 ms per fold for classical fits), with no accuracy gain in return.
- Kernel bandwidth matters. At small bandwidths the kernel concentrates (off-diagonal values near 1), so bandwidth is tuned by CV together with C.

## Dataset

[Wisconsin Breast Cancer (Diagnostic)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), loaded from scikit-learn.

- 569 samples, 30 numeric features, no missing values
- 212 malignant (37.3%), 357 benign (62.7%)
- Labels are flipped so that **malignant = 1** and is the positive class for recall, F1 and ROC-AUC

## Method

### Data split
- Stratified 80/20 train/test split: 455 train rows, 114 test rows
- Inside the training set, 5-fold stratified CV repeated 3 times (**15 folds**)
- All random seeds are fixed

### Preprocessing
Every fitted step is fitted on the training rows of a fold only, then applied to the validation or test rows, so there is no leakage.

1. Yeo-Johnson power transform and standardisation of all 30 features (removes skew: 24 skewed features before, 0 after)
2. Reduce to **4 features** (one per qubit) using one of three representations
3. MinMax scale to `[0, π]` for use as rotation angles (out-of-range values are clipped)

| Representation | Description |
|---|---|
| **PCA-4** | First 4 principal components (about 81% of variance) |
| **MI-4** | Top 4 features by mutual information with the label |
| **MI-decorr-4** | Top 4 by mutual information, skipping any feature with \|r\| > 0.9 against an already kept one |

A fourth representation, **All 30** features, is used for the classical baselines only.

### Quantum feature maps (PennyLane, `default.qubit`, 4 qubits)

- **Entangled angle map:** 2 repetitions of RY and RZ encoding on each qubit, followed by a ring of CNOTs
- **ZZ map:** 2 repetitions of Hadamard, RZ(2x), and pairwise CNOT-RZ-CNOT entanglers

The kernel is the exact fidelity kernel `K(x, z) = |⟨φ(x)|φ(z)⟩|²`, computed from simulated statevectors and checked for symmetry, unit diagonal, value range, positive semi-definiteness and agreement with the slower `U(x)†U(z)` circuit.

### Experiments

| ID | Representation | Feature map |
|---|---|---|
| E1 | PCA-4 | Entangled angle |
| E2 | PCA-4 | ZZ |
| E3 | MI-4 | Entangled angle |
| E4 | MI-4 | ZZ |
| E5 | MI-4 decorr | Entangled angle |
| E6 | MI-4 decorr | ZZ |

### Models and tuning
- **QSVM:** SVM with precomputed kernel, balanced class weights. C ∈ {0.01, 0.1, 1, 10, 100} and bandwidth ∈ {0.25, 0.5, 1.0} are chosen by mean validation ROC-AUC over the 15 folds.
- **RBF SVM:** C ∈ {0.01, 0.1, 1, 10, 100}, gamma ∈ {0.001, 0.01, 0.1, 1, 10}
- **Logistic regression:** C grid as above
- The selected model is refit on the full training set and scored **once** on the held-out test set.

### Statistics
- CV results: mean over 15 folds with Nadeau-Bengio corrected 95% CIs
- Test results: stratified-bootstrap 95% CIs
- Paired comparisons: corrected resampled t-test (CV) and exact McNemar (test), Holm-adjusted

## Results summary (ROC-AUC)

| Model | Representation | CV | Test |
|---|---|---|---|
| QSVM entangled-angle | PCA-4 | 0.993 | 0.991 |
| QSVM entangled-angle | MI-4 | 0.985 | 0.987 |
| QSVM entangled-angle | MI-4 decorr | 0.990 | 0.982 |
| QSVM ZZ | PCA-4 | 0.992 | 0.991 |
| QSVM ZZ | MI-4 | 0.983 | 0.988 |
| QSVM ZZ | MI-4 decorr | 0.988 | 0.982 |
| RBF SVM | PCA-4 | 0.993 | 0.993 |
| RBF SVM | MI-4 | 0.986 | 0.988 |
| RBF SVM | MI-4 decorr | 0.990 | 0.982 |
| RBF SVM | All 30 | 0.996 | 0.993 |
| Logistic regression | PCA-4 | 0.992 | 0.991 |
| Logistic regression | MI-4 | 0.984 | 0.988 |
| Logistic regression | MI-4 decorr | 0.989 | 0.984 |
| Logistic regression | All 30 | 0.996 | 0.992 |

Full tables with recall, F1, accuracy, CIs and runtime are produced by the notebook.

## Getting started

### Requirements
- Python 3.13 (the reference run used 3.13.18)
- `scikit-learn==1.9.1`
- `pennylane==0.45.1`
- numpy 2.4.6, pandas 2.2.3, scipy 1.16.3, matplotlib

The notebook installs and version-checks scikit-learn and PennyLane in its first cell, and asserts the exact versions.

### Running on Google Colab (recommended)
1. Upload the notebook to Colab
2. **Runtime > Run all**
3. Wait about **30 minutes**, mostly spent building quantum kernels

No Google Drive access is needed. Files are written to the session's temporary disk (`DATA_DIR`, default `/content/sc4055_data`), so a new session rebuilds everything. To persist outputs, set `USE_DRIVE = True` in the setup cell.

### Running locally
```bash
pip install scikit-learn==1.9.1 pennylane==0.45.1 pandas scipy matplotlib jupyter
jupyter notebook "quantum computing.ipynb"
```
Run all cells. Set `DATA_DIR_OVERRIDE` in the setup cell if you want outputs in a specific folder.

## Notebook structure

1. **Setup and data preprocessing:** version checks, `preprocessing.py` generation, reference-build comparison, feature selection frequency, data summary, PCA explained variance
2. **Quantum kernel + SVM:** feature maps, circuit diagrams, kernel construction and caching, tuning, final predictions
3. **Classical baselines:** RBF SVM and logistic regression on the same arrays and folds
4. **Results and analysis:** integrated tables, paired statistical tests, kernel concentration, figures

## Outputs

Written to `DATA_DIR`:

| File | Contents |
|---|---|
| `preprocessing.py` | Generated preprocessing module |
| `arrays.npz`, `manifest.json` | Preprocessed arrays for every fold and the final split, with hashes and versions |
| `kernel_cache/` | Cached quantum kernels |
| `quantum_results.csv` | QSVM results |
| `classical_results.csv` | RBF SVM and logistic regression results |
| `p4_*.csv` | Tables 1 to 4 (integrated results, paired comparisons, interaction tests, bandwidth analysis) |
| `pca_explained_variance.png` | PCA scree plot |
| `figures/p4_fig1` to `p4_fig5` | Model by representation, paired comparisons, bandwidth, runtime, fold variability |

## Reproducibility

- All random states are fixed
- Preprocessed arrays are checked against reference summaries from the 2 Nov 2026 run (tolerance 1e-5; differences around 1e-7 between machines come from scipy's optimiser)
- `arrays.npz` is hashed and the library versions are stored in `manifest.json`

## Team

| Section | Member |
|---|---|
| Setup and data preprocessing | Laksh |
| Quantum kernel + SVM | Shanshan |
| Classical baselines | Narue |
| Results and analysis | Siri |

## Limitations

- Quantum kernels are **simulated** exactly on a classical machine. No noise, shot error or real hardware is involved, and the simulation cost is why QSVM is slow here.
- Only 4 qubits and 4 features are used, so the quantum models see far less information than the All-30 classical baselines.
- Several selected hyperparameters sit at the edge of the grid (C = 100, or bandwidth = 0.05), so slightly better settings may exist.
- One dataset and one train/test split. The test set has only 114 rows (42 malignant), so test confidence intervals are wide.
