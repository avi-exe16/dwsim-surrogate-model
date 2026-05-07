
Surrogate Modeling of Benzene-Toluene Distillation


Author  : Abhishek Shandilya
System  : Benzene - Toluene
Model   : Peng-Robinson (PR)


FOLDER STRUCTURE

Surrogate_Model DWSIM/
  |
  |-- DWSIM_Column.dwsim.dwxmz      DWSIM flowsheet file
  |-- distillation_dataset.csv      Generated dataset (2923 rows)
  |-- Report.pdf                    Full project report
  |-- Results_Summary.txt           Model results summary
  |-- README.txt                    This file
  |
  |-- Code/
  |    |-- 01_data_generation.py    DWSIM automation script
  |    |-- 02_model_training.py     ML training pipeline
  |
  |-- Models/
  |    |-- best_model.pkl           Best model (Gradient Boosting)
  |    |-- scaler.pkl               StandardScaler
  |
  |-- Plots/
       |-- distributions.png
       |-- correlation.png
       |-- model_comparison.png
       |-- r2_heatmap.png
       |-- predicted_vs_actual.png
       |-- feature_importance.png
       |-- physical_consistency.png
       |-- learning_curve.png


HOW TO RUN


STEP 1: Open DWSIM Flowsheet
  - Install DWSIM from https://dwsim.org
  - Open DWSIM_Column.dwsim.dwxmz in DWSIM
  - Verify the flowsheet solves (0 errors)
  - Close DWSIM before running automation scripts

STEP 2: Setup Python Environment
  conda create -n dwsim_env python=3.12
  conda activate dwsim_env
  pip install pandas numpy scikit-learn xgboost
              joblib seaborn matplotlib pythonnet

STEP 3: Generate Dataset
  conda activate dwsim_env
  jupyter notebook
  Run: 01_data_generation.py
  Output: distillation_dataset.csv

STEP 4: Train Models
  Switch kernel to: Python 3 (ipykernel)
  Run: 02_model_training.py
  Output: best_model.pkl, all plots, Results_Summary.txt

STEP 5: Make Predictions
  import joblib, numpy as np
  model  = joblib.load('best_model.pkl')
  inputs = np.array([[365, 0.5, 15, 7, 2.5, 50]])
  pred   = model.predict(inputs)
  print(pred)  # [xD, xB, QC, QR]


ASSUMPTIONS


1. Feed molar flow fixed at 100 mol/s
2. Column pressure fixed at 101325 Pa (1 atm)
3. Total condenser assumed
4. Stage efficiency not modeled (ideal stages)
5. Feed stage fraction between 0.25 and 0.65 of total stages


NOTES


- DWSIM must be closed before running data generation script
- Data generation uses DWSIM Python 3.12 kernel
- Model training uses standard Python 3 kernel
- 77 out of 3000 simulations failed to converge and were removed


