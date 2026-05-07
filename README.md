# Surrogate Modeling of Benzene-Toluene Distillation Column

**Author:** Abhishek Shandilya  
**System:** Benzene - Toluene  
**Thermodynamic Model:** Peng-Robinson (PR)

---

## Overview

This project develops a machine learning surrogate model for a binary distillation column separating Benzene and Toluene. The surrogate model is trained on data generated from DWSIM rigorous simulations and predicts key column performance variables from operating conditions.

---

## Inputs and Outputs

### Input Variables
| Variable | Symbol | Unit |
|---|---|---|
| Feed Temperature | T_feed_K | K |
| Feed Pressure | P_feed_Pa | Pa |
| Feed Composition | xF_benzene | mol fraction |
| Number of Stages | N_stages | — |
| Feed Stage Location | feed_stage | — |
| Reflux Ratio | reflux_ratio | — |
| Bottoms Flow Rate | bottoms_flow_mols | mol/s |

### Output Variables
| Variable | Symbol | Unit |
|---|---|---|
| Distillate Purity | xD_benzene | mol fraction |
| Bottoms Purity | xB_toluene | mol fraction |
| Condenser Duty | QC_kW | kW |
| Reboiler Duty | QR_kW | kW |

---

## Dataset

- Sampling method: Latin Hypercube Sampling (LHS)
- Simulations attempted: 3000
- Successful simulations: 2923
- Dataset size: 2923 rows x 11 columns

---

## Models Trained and Compared

| Model | Mean R2 | Grade |
|---|---|---|
| Gradient Boosting | 0.9955 | Excellent |
| Neural Network (MLP) | 0.9937 | Excellent |
| Random Forest | 0.9903 | Excellent |
| SVM (RBF) | 0.9903 | Excellent |
| XGBoost | 0.9876 | Good |
| Polynomial Regression (d=2) | 0.9670 | Good |
| Linear Regression | 0.8765 | Fair |

**Best Model: Gradient Boosting (Mean R2 = 0.9955)**

---

## Best Model Performance (Gradient Boosting)

| Output | R2 | RMSE | MAE |
|---|---|---|---|
| xD_benzene | 0.9962 | 0.0127 | 0.0074 |
| xB_toluene | 0.9967 | 0.0116 | 0.0076 |
| QC_kW | 0.9988 | 0.0611 | 0.0480 |
| QR_kW | 0.9904 | 0.2380 | 0.1611 |

---

## Repository Structure
