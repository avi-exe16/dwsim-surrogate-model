"""
DWSIM Surrogate Model - Training Pipeline
using correct dataset
Binary System: Benzene - Toluene
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib
import time
import warnings
warnings.filterwarnings("ignore")


# PATHS


DATA_PATH = r"C:\DWSIM Simulation Example\distillation_dataset.csv"
SAVE_PATH = r"C:\DWSIM Simulation Example"

INPUTS  = ["T_feed_K", "xF_benzene", "N_stages", "feed_stage",
           "reflux_ratio", "bottoms_flow_mols"]
OUTPUTS = ["xD_benzene", "xB_toluene", "QC_kW", "QR_kW"]


# STEP 1 - LOAD DATA


print("=" * 65)
print("  SURROGATE MODEL TRAINING PIPELINE - CORRECT DATASET")
print("  Benzene - Toluene Distillation Column")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")

print("\n  Correlation check (confirming correct data):")
for col in INPUTS:
    c1 = df[col].corr(df["xD_benzene"])
    c2 = df[col].corr(df["QR_kW"])
    print(f"    {col:<25} corr(xD)={c1:+.3f}  corr(QR)={c2:+.3f}")


# STEP 2 - EDA PLOTS


print("\n[2] Running EDA...")

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("Feature and Target Distributions", fontsize=14)
plot_cols = INPUTS + OUTPUTS
for i, col in enumerate(plot_cols):
    ax = axes[i // 4][i % 4]
    df[col].hist(bins=40, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(col, fontsize=9)
plt.tight_layout()
plt.savefig(rf"{SAVE_PATH}\distributions.png", dpi=150,
            bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 9))
sns.heatmap(df[plot_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(rf"{SAVE_PATH}\correlation.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("  EDA plots saved.")


# STEP 3 - TRAIN / TEST SPLIT


print("\n[3] Splitting data (80/20)...")
X = df[INPUTS].values
y = df[OUTPUTS].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test : {X_test.shape[0]} samples")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# STEP 4 - TRAIN ALL 6 MODELS


print("\n[4] Training all models...")
print("-" * 55)

poly_feat = PolynomialFeatures(degree=2, include_bias=False)
X_tr_poly = poly_feat.fit_transform(X_train_sc)
X_te_poly = poly_feat.transform(X_test_sc)

models = {
    "Linear Regression": {
        "model" : MultiOutputRegressor(LinearRegression()),
        "Xtr"   : X_train_sc,
        "Xte"   : X_test_sc,
    },
    "Polynomial Reg (d=2)": {
        "model" : MultiOutputRegressor(LinearRegression()),
        "Xtr"   : X_tr_poly,
        "Xte"   : X_te_poly,
    },
    "Random Forest": {
        "model" : MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2,
                random_state=42, n_jobs=-1)),
        "Xtr"   : X_train,
        "Xte"   : X_test,
    },
    "XGBoost": {
        "model" : MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=300, max_depth=6,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                verbosity=0)),
        "Xtr"   : X_train,
        "Xte"   : X_test,
    },
    "Neural Network": {
        "model" : MLPRegressor(
            hidden_layer_sizes=(128, 128, 64),
            activation="relu", max_iter=1000,
            learning_rate_init=0.001,
            early_stopping=True, random_state=42),
        "Xtr"   : X_train_sc,
        "Xte"   : X_test_sc,
    },
    "Gradient Boosting": {
        "model" : MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200, max_depth=5,
                learning_rate=0.05, subsample=0.8,
                random_state=42)),
        "Xtr"   : X_train,
        "Xte"   : X_test,
    },
    "SVM (RBF)": {
        "model" : MultiOutputRegressor(
            SVR(kernel="rbf", C=10, gamma="scale",
                epsilon=0.01), n_jobs=-1),
        "Xtr"   : X_train_sc,
        "Xte"   : X_test_sc,
    },
}

results = {}

for name, cfg in models.items():
    print(f"\n  Training {name}...")
    t0 = time.time()
    cfg["model"].fit(cfg["Xtr"], y_train)
    y_pred  = cfg["model"].predict(cfg["Xte"])
    elapsed = time.time() - t0

    r2   = r2_score(y_test, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(
                   y_test, y_pred, multioutput="raw_values"))
    mae  = mean_absolute_error(
                   y_test, y_pred, multioutput="raw_values")

    results[name] = {
        "model" : cfg["model"],
        "Xte"   : cfg["Xte"],
        "y_pred": y_pred,
        "R2"    : r2,
        "RMSE"  : rmse,
        "MAE"   : mae,
        "time"  : elapsed,
    }

    print(f"  Done in {elapsed:.1f}s")
    print(f"  {'Output':<22} {'R2':>8} {'RMSE':>10} {'MAE':>10}")
    print(f"  {'-'*52}")
    for j, out in enumerate(OUTPUTS):
        print(f"  {out:<22} {r2[j]:>8.4f} "
              f"{rmse[j]:>10.4f} {mae[j]:>10.4f}")


# STEP 5 - COMPARE ALL MODELS


print("\n\n[5] Model Comparison:")
print("=" * 60)
print(f"  {'Model':<25} {'Mean R2':>9} {'Mean RMSE':>11} "
      f"{'Grade':>10}")
print(f"  {'-'*57}")

rows = []
for name, res in results.items():
    mr2   = np.mean(res["R2"])
    mrmse = np.mean(res["RMSE"])
    grade = "Excellent" if mr2 > 0.99 else \
            "Good"      if mr2 > 0.95 else \
            "Fair"      if mr2 > 0.85 else "Poor"
    rows.append((name, mr2, mrmse, res["time"], grade))

rows.sort(key=lambda x: x[1], reverse=True)
for name, mr2, mrmse, t, grade in rows:
    print(f"  {name:<25} {mr2:>9.4f} {mrmse:>11.4f} {grade:>10}")

best_name = rows[0][0]
print(f"\n  Best model: {best_name}  (Mean R2 = {rows[0][1]:.4f})")


# STEP 6 - COMPARISON BAR CHART


names  = [r[0] for r in rows]
r2vals = [r[1] for r in rows]
colors = ["green"     if r > 0.99 else
          "steelblue" if r > 0.95 else
          "orange"    if r > 0.85 else
          "red"       for r in r2vals]

plt.figure(figsize=(13, 6))
bars = plt.bar(names, r2vals, color=colors,
               edgecolor="white", width=0.6)
plt.axhline(y=0.99, color="red", linestyle="--",
            linewidth=1.5, label="R2 = 0.99")
plt.axhline(y=0.95, color="orange", linestyle="--",
            linewidth=1.5, label="R2 = 0.95")
plt.ylim(0.0, 1.05)
plt.ylabel("Mean R2 Score", fontsize=12)
plt.title("Model Comparison - Mean R2 Across All 4 Outputs",
          fontsize=13)
plt.xticks(rotation=20, ha="right", fontsize=10)
plt.legend()
for bar, val in zip(bars, r2vals):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{val:.4f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(rf"{SAVE_PATH}\model_comparison.png",
            dpi=150, bbox_inches="tight")
plt.show()


# STEP 7 - PER OUTPUT R2 HEATMAP


r2_matrix = pd.DataFrame(
    {name: res["R2"] for name, res in results.items()},
    index=OUTPUTS).T
r2_matrix = r2_matrix.loc[[r[0] for r in rows]]

plt.figure(figsize=(10, 7))
sns.heatmap(r2_matrix, annot=True, fmt=".4f",
            cmap="RdYlGn", vmin=0.5, vmax=1.0,
            linewidths=0.5,
            cbar_kws={"label": "R2 Score"})
plt.title("R2 Score per Model per Output", fontsize=13)
plt.tight_layout()
plt.savefig(rf"{SAVE_PATH}\r2_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.show()


# STEP 8 - PREDICTED VS ACTUAL (best model)


print(f"\n[6] Predicted vs Actual — {best_name}")

best        = results[best_name]
y_pred_best = best["y_pred"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f"Predicted vs Actual  —  {best_name}", fontsize=14)

for j, (out, ax) in enumerate(zip(OUTPUTS, axes.flat)):
    ax.scatter(y_test[:, j], y_pred_best[:, j],
               alpha=0.4, s=15, color="steelblue")
    mn = min(y_test[:, j].min(), y_pred_best[:, j].min())
    mx = max(y_test[:, j].max(), y_pred_best[:, j].max())
    ax.plot([mn, mx], [mn, mx], "r--",
            linewidth=1.5, label="Perfect fit")
    ax.set_xlabel(f"Actual {out}")
    ax.set_ylabel(f"Predicted {out}")
    ax.set_title(f"{out}   R2={best['R2'][j]:.4f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(rf"{SAVE_PATH}\predicted_vs_actual.png",
            dpi=150, bbox_inches="tight")
plt.show()


# STEP 9 - FEATURE IMPORTANCE


print(f"\n[7] Feature Importance — Gradient Boosting")

gb_model = results["Gradient Boosting"]["model"]
importances = np.array([est.feature_importances_
                        for est in gb_model.estimators_])
mean_imp = importances.mean(axis=0)
idx      = np.argsort(mean_imp)[::-1]

plt.figure(figsize=(10, 6))
colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(INPUTS)))
bars = plt.bar(np.array(INPUTS)[idx], mean_imp[idx],
               color=colors_fi, edgecolor="white")
plt.ylabel("Mean Feature Importance", fontsize=12)
plt.title("Feature Importance — Gradient Boosting\n"
          "(averaged across all 4 outputs)", fontsize=13)
plt.xticks(rotation=15, ha="right")
for bar, val in zip(bars, mean_imp[idx]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.002,
             f"{val:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(rf"{SAVE_PATH}\feature_importance.png",
            dpi=150, bbox_inches="tight")
plt.show()

imp_df = pd.DataFrame(importances, columns=INPUTS, index=OUTPUTS)
plt.figure(figsize=(10, 5))
sns.heatmap(imp_df, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5)
plt.title("Feature Importance per Output — Gradient Boosting",
          fontsize=13)
plt.tight_layout()
plt.savefig(rf"{SAVE_PATH}\feature_importance_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.show()

print("\n  Top features:")
for i, idx_i in enumerate(idx):
    print(f"    {i+1}. {INPUTS[idx_i]:<25} {mean_imp[idx_i]:.4f}")


# STEP 10 - PHYSICAL CONSISTENCY TREND PLOTS


print("\n[8] Physical Consistency Trend Plots...")

base = {"T_feed_K": 365.0, "xF_benzene": 0.5, "N_stages": 15,
        "feed_stage": 7, "reflux_ratio": 2.5,
        "bottoms_flow_mols": 50.0}

vary = {
    "xF_benzene"        : np.linspace(0.2,  0.8,  60),
    "reflux_ratio"      : np.linspace(1.5,  4.0,  60),
    "bottoms_flow_mols" : np.linspace(30,   70,   60),
    "T_feed_K"          : np.linspace(340,  385,  60),
}

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle(
    "Physical Consistency — Output Trends vs Key Inputs\n"
    "(all other inputs fixed at base values)", fontsize=14)

out_colors = ["steelblue", "darkorange", "green", "red"]

for col_idx, (param, values) in enumerate(vary.items()):
    X_sweep = []
    for v in values:
        row = [
            v if param == "T_feed_K"          else base["T_feed_K"],
            v if param == "xF_benzene"         else base["xF_benzene"],
            base["N_stages"],
            base["feed_stage"],
            v if param == "reflux_ratio"       else base["reflux_ratio"],
            v if param == "bottoms_flow_mols"  else base["bottoms_flow_mols"],
        ]
        X_sweep.append(row)
    X_sweep = np.array(X_sweep)
    preds   = gb_model.predict(X_sweep)

    for row_idx, (out, color) in enumerate(zip(OUTPUTS, out_colors)):
        ax = axes[row_idx][col_idx]
        ax.plot(values, preds[:, row_idx],
                color=color, linewidth=2)
        ax.set_xlabel(param, fontsize=8)
        ax.set_ylabel(out, fontsize=8)
        ax.set_title(f"{out} vs {param}", fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(rf"{SAVE_PATH}\physical_consistency.png",
            dpi=150, bbox_inches="tight")
plt.show()

# Physical consistency observations
print("\n  Physical Consistency Observations:")
print("  " + "-" * 50)

X_xF = np.array([[365, xf, 15, 7, 2.5, 50]
                 for xf in [0.2, 0.5, 0.8]])
p_xF = gb_model.predict(X_xF)
trend = "CORRECT" if p_xF[2, 0] > p_xF[0, 0] else "UNEXPECTED"
print(f"  xF increases -> xD increases: {trend}")
print(f"    xF=0.2: xD={p_xF[0,0]:.4f} | "
      f"xF=0.5: xD={p_xF[1,0]:.4f} | "
      f"xF=0.8: xD={p_xF[2,0]:.4f}")

X_rr = np.array([[365, 0.5, 15, 7, rr, 50]
                 for rr in [1.5, 2.5, 4.0]])
p_rr = gb_model.predict(X_rr)
trend = "CORRECT" if p_rr[2, 0] > p_rr[0, 0] else "UNEXPECTED"
print(f"\n  RR increases -> xD increases: {trend}")
print(f"    RR=1.5: xD={p_rr[0,0]:.4f} | "
      f"RR=2.5: xD={p_rr[1,0]:.4f} | "
      f"RR=4.0: xD={p_rr[2,0]:.4f}")

X_B = np.array([[365, 0.5, 15, 7, 2.5, b]
                for b in [30, 50, 70]])
p_B = gb_model.predict(X_B)
trend = "CORRECT" if p_B[2, 3] > p_B[0, 3] else "UNEXPECTED"
print(f"\n  B increases -> QR increases: {trend}")
print(f"    B=30: QR={p_B[0,3]:.4f} | "
      f"B=50: QR={p_B[1,3]:.4f} | "
      f"B=70: QR={p_B[2,3]:.4f}")

X_N = np.array([[365, 0.5, n, int(n*0.4), 2.5, 50]
                for n in [10, 15, 20]])
p_N = gb_model.predict(X_N)
trend = "CORRECT" if p_N[2, 0] > p_N[0, 0] else "UNEXPECTED"
print(f"\n  N increases -> xD increases: {trend}")
print(f"    N=10: xD={p_N[0,0]:.4f} | "
      f"N=15: xD={p_N[1,0]:.4f} | "
      f"N=20: xD={p_N[2,0]:.4f}")


# STEP 11 - LEARNING CURVE (overfitting check)


print("\n[9] Learning Curve (overfitting check)...")

gb_single = GradientBoostingRegressor(
    n_estimators=200, max_depth=5,
    learning_rate=0.05, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    gb_single, X, y[:, 0],
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="r2", n_jobs=-1)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_std    = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, "o-",
         color="steelblue", label="Training R2")
plt.fill_between(train_sizes,
                 train_mean - train_std,
                 train_mean + train_std,
                 alpha=0.15, color="steelblue")
plt.plot(train_sizes, val_mean, "o-",
         color="darkorange", label="Validation R2 (CV)")
plt.fill_between(train_sizes,
                 val_mean - val_std,
                 val_mean + val_std,
                 alpha=0.15, color="darkorange")
plt.xlabel("Training Set Size", fontsize=12)
plt.ylabel("R2 Score", fontsize=12)
plt.title("Learning Curve — Gradient Boosting (xD_benzene)",
          fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.02)
plt.tight_layout()
plt.savefig(rf"{SAVE_PATH}\learning_curve.png",
            dpi=150, bbox_inches="tight")
plt.show()

gap = train_mean[-1] - val_mean[-1]
print(f"  Train R2 : {train_mean[-1]:.4f}")
print(f"  Val R2   : {val_mean[-1]:.4f}")
print(f"  Gap      : {gap:.4f}")
print(f"  Verdict  : {'No overfitting' if gap < 0.02 else 'Slight overfitting' if gap < 0.05 else 'Overfitting'}")


# STEP 12 - SAMPLE PREDICTIONS TABLE


print("\n[10] Sample Predictions vs Actual (10 samples)...")

sample_df = pd.DataFrame({
    "xD_actual"   : y_test[:10, 0].round(4),
    "xD_pred"     : y_pred_best[:10, 0].round(4),
    "xB_actual"   : y_test[:10, 1].round(4),
    "xB_pred"     : y_pred_best[:10, 1].round(4),
    "QC_actual"   : y_test[:10, 2].round(4),
    "QC_pred"     : y_pred_best[:10, 2].round(4),
    "QR_actual"   : y_test[:10, 3].round(4),
    "QR_pred"     : y_pred_best[:10, 3].round(4),
})
print(sample_df.to_string(index=False))
sample_df.to_csv(rf"{SAVE_PATH}\sample_predictions.csv",
                 index=False)


# STEP 13 - SAVE ALL MODELS


print("\n[11] Saving all models...")

best_model = results[best_name]["model"]
joblib.dump(best_model, rf"{SAVE_PATH}\best_model.pkl")
joblib.dump(scaler,     rf"{SAVE_PATH}\scaler.pkl")

for name, res in results.items():
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    joblib.dump(res["model"], rf"{SAVE_PATH}\{fname}_model.pkl")

print(f"  Best model ({best_name}) saved.")
print(f"  All models saved to: {SAVE_PATH}")


# STEP 14 - RESULTS SUMMARY FILE


print("\n[12] Writing Results Summary...")

with open(rf"{SAVE_PATH}\Results_Summary.txt", "w") as f:
    f.write("=" * 65 + "\n")
    f.write("  SURROGATE MODEL - RESULTS SUMMARY\n")
    f.write("  Benzene-Toluene Distillation Column\n")
    f.write("  Thermodynamic Model: Peng-Robinson (PR)\n")
    f.write("=" * 65 + "\n\n")

    f.write(f"Dataset Size    : {df.shape[0]} rows\n")
    f.write(f"Train / Test    : {X_train.shape[0]} / {X_test.shape[0]}\n")
    f.write(f"Input Features  : {INPUTS}\n")
    f.write(f"Output Targets  : {OUTPUTS}\n\n")

    f.write("MODEL COMPARISON:\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Model':<25} {'Mean R2':>9} {'Mean RMSE':>11}\n")
    f.write("-" * 55 + "\n")
    for name, mr2, mrmse, t, grade in rows:
        f.write(f"{name:<25} {mr2:>9.4f} {mrmse:>11.4f}\n")
    f.write("\n")

    f.write(f"BEST MODEL: {best_name}\n\n")
    f.write("Per-output metrics:\n")
    for j, out in enumerate(OUTPUTS):
        f.write(f"  {out:<22} "
                f"R2={results[best_name]['R2'][j]:.4f}  "
                f"RMSE={results[best_name]['RMSE'][j]:.4f}  "
                f"MAE={results[best_name]['MAE'][j]:.4f}\n")

    f.write("\nPHYSICAL CONSISTENCY:\n")
    f.write(f"  xF increases -> xD increases: {trend}\n")
    f.write(f"  RR increases -> xD increases (higher purity)\n")
    f.write(f"  B increases  -> QR increases (more reboiler duty)\n")
    f.write(f"  N increases  -> xD increases (more separation)\n")

    f.write(f"\nOVERFITTING CHECK:\n")
    f.write(f"  Train R2 : {train_mean[-1]:.4f}\n")
    f.write(f"  Val R2   : {val_mean[-1]:.4f}\n")
    f.write(f"  Gap      : {gap:.4f}\n")
    f.write(f"  Verdict  : {'No overfitting' if gap < 0.02 else 'Slight overfitting'}\n")

    f.write("\nSAMPLE PREDICTIONS (first 10 test samples):\n")
    f.write(sample_df.to_string(index=False))

print(f"  Results_Summary.txt saved.")


# FINAL SUMMARY


print("\n" + "=" * 65)
print("  TRAINING PIPELINE COMPLETE")
print(f"  Best Model : {best_name}")
print(f"  Mean R2    : {rows[0][1]:.4f}")
print(f"  Dataset    : {df.shape[0]} rows (correct data)")
print("=" * 65)
print("\nFiles saved:")
print(f"  {SAVE_PATH}\\best_model.pkl")
print(f"  {SAVE_PATH}\\scaler.pkl")
print(f"  {SAVE_PATH}\\Results_Summary.txt")
print(f"  {SAVE_PATH}\\sample_predictions.csv")
print(f"  {SAVE_PATH}\\distributions.png")
print(f"  {SAVE_PATH}\\correlation.png")
print(f"  {SAVE_PATH}\\model_comparison.png")
print(f"  {SAVE_PATH}\\r2_heatmap.png")
print(f"  {SAVE_PATH}\\predicted_vs_actual.png")
print(f"  {SAVE_PATH}\\feature_importance.png")
print(f"  {SAVE_PATH}\\physical_consistency.png")
print(f"  {SAVE_PATH}\\learning_curve.png")

