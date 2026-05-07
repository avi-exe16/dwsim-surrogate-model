# Surrogate Modeling of Benzene-Toluene Distillation Column Using DWSIM and Machine Learning

**Author:** Abhishek Shandilya  
**System:** Benzene - Toluene  
**Thermodynamic Model:** Peng-Robinson (PR)  
**Best Model:** Gradient Boosting (Mean R2 = 0.9955)

---

## Objective

Develop a machine learning surrogate model for a binary distillation column that predicts key performance variables from operating conditions, replacing computationally expensive DWSIM rigorous simulations.

---

## Input and Output Variables

### Inputs
| Variable | Symbol | Unit |
|---|---|---|
| Feed Temperature | T_feed_K | K |
| Feed Pressure | P_feed_Pa | Pa |
| Feed Composition | xF_benzene | mol fraction |
| Number of Stages | N_stages | — |
| Feed Stage Location | feed_stage | — |
| Reflux Ratio | reflux_ratio | — |
| Bottoms Flow Rate | bottoms_flow_mols | mol/s |

### Outputs
| Variable | Symbol | Unit |
|---|---|---|
| Distillate Purity | xD_benzene | mol fraction |
| Bottoms Purity | xB_toluene | mol fraction |
| Condenser Duty | QC_kW | kW |
| Reboiler Duty | QR_kW | kW |

---

## Dataset

| Parameter | Value |
|---|---|
| Sampling Method | Latin Hypercube Sampling (LHS) |
| Simulations Attempted | 3000 |
| Successful Simulations | 2923 |
| Final Dataset Size | 2923 rows x 11 columns |
| Train / Test Split | 80% / 20% |

---

## Model Comparison

| Model | Mean R2 | Mean RMSE | Grade |
|---|---|---|---|
| Gradient Boosting | 0.9955 | 0.0808 | Excellent |
| Neural Network (MLP) | 0.9937 | 0.0408 | Excellent |
| Random Forest | 0.9903 | 0.1248 | Excellent |
| SVM (RBF) | 0.9903 | 0.1144 | Excellent |
| XGBoost | 0.9876 | 0.1216 | Good |
| Polynomial Regression (d=2) | 0.9670 | 0.1850 | Good |
| Linear Regression | 0.8765 | 0.3107 | Fair |

---

## Best Model Performance — Gradient Boosting

| Output | R2 | RMSE | MAE |
|---|---|---|---|
| xD_benzene | 0.9962 | 0.0127 | 0.0074 |
| xB_toluene | 0.9967 | 0.0116 | 0.0076 |
| QC_kW | 0.9988 | 0.0611 | 0.0480 |
| QR_kW | 0.9904 | 0.2380 | 0.1611 |

**Overfitting Check:** Train R2 = 0.9993, Validation R2 = 0.9956, Gap = 0.0037 — No overfitting detected.

---

## Repository Structure

```
dwsim-surrogate-model/
├── DWSIM_Column.dwsim.dwxmz               DWSIM flowsheet file
├── Results_Summary.txt                    Model results and metrics
├── Results_Summary_Predictions.csv        Sample predictions vs actual
├── Code/
│   ├── 01_data_generation.py              DWSIM automation script
│   └── 02_model_training.py               ML training pipeline
├── Dataset/
│   └── distillation_dataset.csv           Generated dataset (2923 rows)
├── Models/
│   ├── best_model.pkl                     Gradient Boosting (best model)
│   ├── linear_regression_model.pkl        Linear Regression model
│   └── scaler.pkl                         StandardScaler
└── Plots/
    ├── distributions.png
    ├── correlation.png
    ├── model_comparison.png
    ├── r2_heatmap.png
    ├── predicted_vs_actual.png
    ├── feature_importance.png
    ├── feature_importance_heatmap.png
    ├── physical_consistency.png
    └── learning_curve.png
```

---

## How to Run

### 1. Setup Environment

```
conda create -n dwsim_env python=3.12
conda activate dwsim_env
pip install pandas numpy scikit-learn xgboost joblib seaborn matplotlib pythonnet
```

### 2. Generate Dataset
- Install DWSIM from https://dwsim.org
- Open `DWSIM_Column.dwsim.dwxmz` and verify it solves with 0 errors
- Close DWSIM completely before running the script
- Run `Code/01_data_generation.py` using the DWSIM Python 3.12 kernel

### 3. Train Models
- Run `Code/02_model_training.py` using standard Python 3 kernel
- All plots, models, and results are saved automatically

### 4. Make a Prediction

```python
import joblib
import numpy as np

model = joblib.load('Models/best_model.pkl')
inputs = np.array([[365.0, 0.5, 15, 7, 2.5, 50.0]])
pred = model.predict(inputs)

print("xD_benzene:", round(pred[0, 0], 4))
print("xB_toluene:", round(pred[0, 1], 4))
print("QC_kW     :", round(pred[0, 2], 4))
print("QR_kW     :", round(pred[0, 3], 4))
```

---

## Notes

- DWSIM must be fully closed before running the data generation script
- Data generation runs on DWSIM Python 3.12 kernel
- Model training runs on standard Python 3 kernel
- Large model files are excluded due to GitHub file size limits but are generated automatically when the training script is run
