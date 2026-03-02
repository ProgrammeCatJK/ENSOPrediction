# ENSO Niño3.4 Index Forecasting — Assignment 1

## Overview

This project builds and evaluates deep learning models to forecast the **Niño3.4 sea surface temperature index** up to 6 months ahead, using historical lag features as inputs. The work is split across two tasks: individual single-output models (Task A) and a unified multi-output model (Task B).

---

## Dataset

- **`Nino3.4_data.csv`** — Full historical Niño3.4 dataset with lag features (`nino_tminus2`, `nino_tminus1`, `nino_t`) and targets (`nino_tplus1` through `nino_tplus6`).
- **`test_years.csv`** — Predefined test years kept separate from all model training and validation.

---

## Data Preprocessing

- Test years are extracted first and held out completely.
- Remaining data is randomly split into **training (67%)** and **validation (33%)** sets using a fixed random seed (`69`) for reproducibility.
- Input features (`nino_tminus2`, `nino_tminus1`, `nino_t`) are scaled using **Min-Max normalisation** — scaler is fit on training data only and applied to validation/test sets.
- Scalers are saved via `joblib` for consistent inference.

---

## Task A — Single-Output Models (`z5507377_Assigment_1A.ipynb`)

Six separate feedforward neural networks are trained, one per forecast horizon:

| Model | Target |
|-------|--------|
| `nino_model1.keras` | t+1 |
| `nino_model2.keras` | t+2 |
| `nino_model3.keras` | t+3 |
| `nino_model4.keras` | t+4 |
| `nino_model5.keras` | t+5 |
| `nino_model6.keras` | t+6 |

**Architecture:** `Dense(8, tanh) → Dense(1)`  
**Optimiser:** Adam | **Loss:** MSE | **Early Stopping:** patience=10 on val_loss

### Transfer Learning
Model trained on t+1 is fine-tuned sequentially for t+2 through t+6, reusing learned weights to improve convergence on longer horizons. Saved as `Transfered_model_tplus{i}.keras`.

---

## Task B — Multi-Output Model (`z5507377_Assigment_1B.ipynb`)

A single model predicts all 6 horizons simultaneously.

**Architecture:** `Dense(16, tanh) → Dense(8, tanh) → Dense(8, tanh) → Dense(6)`  
**Optimiser:** Adam (lr=0.001) | **Loss:** MSE / Weighted MSE | **Early Stopping:** patience=10

### Weighted Loss Variant
A custom weighted MSE loss prioritises near-term accuracy:

```
weights = [0.30, 0.25, 0.20, 0.15, 0.07, 0.03]  # t+1 → t+6
```

Saved as `Weighted_MOmodel.keras`.

---

## Evaluation (`z5507377_Evaluation.ipynb`)

All models are evaluated on the held-out test set using:
- **RMSE** (Root Mean Squared Error)
- **Pearson Correlation Coefficient**

Results are saved to:
- `TaskA_Evaluation.csv`
- `TaskB_Evaluation.csv`
- `TaskB_Weighted_Evaluation.csv`
- `Combined_Evaluation_Table.csv`

Comparison plots (RMSE and Pearson correlation across horizons) are saved as:
- `RMSE_Comparison.png`
- `Correlation_Comparison.png`

---

## Dependencies

```
tensorflow / keras
scikit-learn
pandas
numpy
matplotlib
joblib
```

---

## File Structure

```
├── Nino3.4_data.csv               # Raw dataset
├── test_years.csv                 # Held-out test years
├── trainingSet.csv / trainingSetB.csv
├── validationSet.csv / validationSetB.csv
├── testSet.csv / testSetB.csv
├── input_scaler.pkl / input_scalerB.pkl
├── nino_model1–6.keras            # Task A single-output models
├── Transfered_model_tplus2–6.keras # Transfer learning models
├── MOmodel.keras                  # Task B multi-output model
├── Weighted_MOmodel.keras         # Task B weighted loss model
├── z5507377_Assigment_1A.ipynb
├── z5507377_Assigment_1B.ipynb
└── z5507377_Evaluation.ipynb
```
