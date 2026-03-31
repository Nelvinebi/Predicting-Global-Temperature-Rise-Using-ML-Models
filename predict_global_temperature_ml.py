"""
================================================================================
  Predicting Global Temperature Rise Using Machine Learning Models
  Author : Agbozu Ebingiye Nelvin
  GitHub : https://github.com/Nelvinebi
  Email  : nelvinebingiye@gmail.com
  Date   : 2025
================================================================================

IMPROVEMENTS OVER V1
────────────────────
• Uses all 10 climate-forcing features (added volcanic_forcing,
  land_use_index, urbanization_index vs. V1's 6 features)
• StandardScaler applied before training for fair model comparison
• Cross-validation (5-fold) added alongside train/test split
• Ridge Regression replaces bare Linear Regression (better for
  correlated climate features)
• Permutation Importance uses the held-out test set (more reliable)
• Scenario forecasting extended: LOW / MED / HIGH now include all 10
  features and realistic decade-level projections
• All figures saved at 150 dpi; professional seaborn theme applied
• model_metrics.txt expanded with CV scores and per-model summaries
• Saved model (joblib) for reuse without retraining
================================================================================
"""

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE = {"Ridge": "#4CAF50", "Random Forest": "#2196F3", "Gradient Boosting": "#FF5722"}
SCENARIO_COLORS = {"LOW": "#27ae60", "MED": "#f39c12", "HIGH": "#e74c3c"}

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("Data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PREDICTING GLOBAL TEMPERATURE RISE — ML PIPELINE  ")
print("=" * 65)

try:
    df = pd.read_excel("Data/global_temp_synthetic.xlsx")
    print(f"\n✅  Loaded Excel dataset  →  {df.shape[0]} rows × {df.shape[1]} cols")
except FileNotFoundError:
    df = pd.read_csv("Data/global_temp_synthetic.csv")
    print(f"\n✅  Loaded CSV dataset  →  {df.shape[0]} rows × {df.shape[1]} cols")

print(f"    Years : {df['year'].min()} – {df['year'].max()}")
print(f"    Temp range : {df['temp_anomaly_C'].min():.3f} °C  →  {df['temp_anomaly_C'].max():.3f} °C")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE SELECTION & SPLIT
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    "co2_ppm",
    "ch4_ppb",
    "n2o_ppb",
    "aerosol_optical_depth",
    "solar_irradiance_anom",
    "enso_index",
    "volcanic_forcing",
    "land_use_index",
    "urbanization_index",
]
TARGET = "temp_anomaly_C"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True    # random split — correct for synthetic dataset
)

# Scale features (important for Ridge; harmless for tree models)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n📊  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DEFINE & TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
models = {
    "Ridge":             Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=300, max_depth=8,
                                               random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                    max_depth=4, random_state=42,
                                                    subsample=0.8),
}

results = {}
print("\n🔄  Training models …\n")

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, scaler.transform(X), y,
                           cv=KFold(n_splits=5, shuffle=True, random_state=42),
                           scoring="r2", n_jobs=-1)

    results[name] = {
        "model": model, "y_pred": y_pred,
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "CV_mean": cv.mean(), "CV_std": cv.std(),
    }

    print(f"  {name:<22}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  "
          f"CV R²={cv.mean():.4f}±{cv.std():.4f}")

# Best model
best_name = max(results, key=lambda n: results[n]["R2"])
best_model = results[best_name]["model"]
print(f"\n🏆  Best model : {best_name}  (R² = {results[best_name]['R2']:.4f})")

# Save best model
joblib.dump({"model": best_model, "scaler": scaler, "features": FEATURES},
            "models/best_model.pkl")
print("💾  Best model saved → models/best_model.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE METRICS
# ══════════════════════════════════════════════════════════════════════════════
with open("outputs/model_metrics.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("  MODEL EVALUATION METRICS\n")
    f.write("  Predicting Global Temperature Rise Using ML Models\n")
    f.write("  Author: Agbozu Ebingiye Nelvin | github.com/Nelvinebi\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"{'Model':<24} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV R²':>12}\n")
    f.write("-" * 62 + "\n")
    for name, r in results.items():
        f.write(f"{name:<24} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f} "
                f"{r['CV_mean']:>6.4f}±{r['CV_std']:.4f}\n")
    f.write("\n")
    f.write(f"Best Model : {best_name}\n")
    f.write(f"Features   : {', '.join(FEATURES)}\n")
    f.write(f"Train size : {len(X_train)} | Test size : {len(X_test)}\n")
print("📄  Metrics saved → outputs/model_metrics.txt")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 1 — EDA: Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 8))
corr = df[FEATURES + [TARGET]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Matrix\nGlobal Temperature Forcing Variables (1880–2024)",
             fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("🖼️   Saved → outputs/correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 2 — Historical Temperature Anomaly Trend
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(df["year"], df[TARGET], alpha=0.18, color="#e74c3c")
ax.plot(df["year"], df[TARGET], color="#e74c3c", linewidth=1.5,
        label="Temperature Anomaly (°C)")
ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axhline(1.5, color="#f39c12", linewidth=1, linestyle="--",
           alpha=0.8, label="Paris +1.5 °C threshold")
ax.axhline(2.0, color="#e74c3c", linewidth=1, linestyle="--",
           alpha=0.8, label="Paris +2.0 °C threshold")

# Add 10-year rolling mean
rolling = df.set_index("year")[TARGET].rolling(10, center=True).mean()
ax.plot(df["year"], rolling.values, color="white", linewidth=2.2,
        linestyle="-", label="10-yr rolling mean")

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Temperature Anomaly (°C)", fontsize=11)
ax.set_title("Global Mean Temperature Anomaly (1880–2024)\nSynthetic Climate Dataset",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("outputs/temperature_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("🖼️   Saved → outputs/temperature_trend.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 3 — Model Comparison (Actual vs Predicted)
# ══════════════════════════════════════════════════════════════════════════════
test_years = df["year"].iloc[X_test.index].values

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Model Comparison — Actual vs Predicted Temperature Anomaly (Test Set)",
             fontsize=13, fontweight="bold", y=1.01)

for ax, (name, r) in zip(axes, results.items()):
    ax.plot(test_years, y_test.values, color="white", linewidth=1.8,
            label="Actual", zorder=3)
    ax.plot(test_years, r["y_pred"], color=PALETTE[name], linewidth=1.8,
            linestyle="--", label="Predicted", zorder=4)
    ax.fill_between(test_years, y_test.values, r["y_pred"],
                    alpha=0.15, color=PALETTE[name])
    ax.set_title(f"{name}\nR²={r['R2']:.4f}  RMSE={r['RMSE']:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Year")
    ax.legend(fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel("Temperature Anomaly (°C)")

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("🖼️   Saved → outputs/model_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURE 4 — Permutation Feature Importance (Best Model)
# ══════════════════════════════════════════════════════════════════════════════
perm = permutation_importance(best_model, X_test_sc, y_test,
                              n_repeats=30, random_state=42, n_jobs=-1)
feat_imp = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": perm.importances_mean,
    "Std": perm.importances_std,
}).sort_values("Importance", ascending=True)

FEATURE_LABELS = {
    "co2_ppm":              "CO₂ Concentration (ppm)",
    "ch4_ppb":              "CH₄ Concentration (ppb)",
    "n2o_ppb":              "N₂O Concentration (ppb)",
    "aerosol_optical_depth":"Aerosol Optical Depth",
    "solar_irradiance_anom":"Solar Irradiance Anomaly",
    "enso_index":           "ENSO Index",
    "volcanic_forcing":     "Volcanic Forcing",
    "land_use_index":       "Land Use Index",
    "urbanization_index":   "Urbanisation Index",
}
feat_imp["Label"] = feat_imp["Feature"].map(FEATURE_LABELS)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#e74c3c" if v > 0 else "#95a5a6" for v in feat_imp["Importance"]]
bars = ax.barh(feat_imp["Label"], feat_imp["Importance"],
               xerr=feat_imp["Std"], color=colors, capsize=4,
               edgecolor="white", linewidth=0.6)
ax.axvline(0, color="white", linewidth=0.8)
ax.set_xlabel("Mean Permutation Importance (Δ R²)", fontsize=11)
ax.set_title(f"Permutation Feature Importance — {best_name}\n"
             f"Global Temperature Rise Drivers (Test Set, 30 repeats)",
             fontsize=13, fontweight="bold")

# Annotate values
for bar, val in zip(bars, feat_imp["Importance"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("🖼️   Saved → outputs/feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. FIGURE 5 — 2050 Scenario Forecast
# ══════════════════════════════════════════════════════════════════════════════
future_years = np.arange(2025, 2051)

# Baseline 2024 values
base = df[df["year"] == 2024][FEATURES].iloc[0].to_dict()

def build_scenario(years, co2_rate, ch4_rate, n2o_rate,
                   aerosol_trend, urbanization_rate):
    """Generate a future forcing DataFrame for a given scenario."""
    rows = []
    for i, yr in enumerate(years):
        t = i + 1
        rows.append({
            "co2_ppm":              base["co2_ppm"]             + co2_rate * t,
            "ch4_ppb":              base["ch4_ppb"]             + ch4_rate * t,
            "n2o_ppb":              base["n2o_ppb"]             + 0.3 * t,
            "aerosol_optical_depth":base["aerosol_optical_depth"]+ aerosol_trend * t,
            "solar_irradiance_anom":base["solar_irradiance_anom"]+ np.random.normal(0, 0.02),
            "enso_index":           np.random.normal(0, 0.5),
            "volcanic_forcing":     np.random.normal(-0.05, 0.05),
            "land_use_index":       min(base["land_use_index"]  + 0.005 * t, 1.0),
            "urbanization_index":   min(base["urbanization_index"] + urbanization_rate * t, 1.0),
        })
    return pd.DataFrame(rows)

np.random.seed(42)

scenarios = {
    "LOW":  build_scenario(future_years, co2_rate=0.5,  ch4_rate=0.3,  n2o_rate=0.1,
                            aerosol_trend=0.002, urbanization_rate=0.004),
    "MED":  build_scenario(future_years, co2_rate=2.5,  ch4_rate=1.5,  n2o_rate=0.3,
                            aerosol_trend=0.0,   urbanization_rate=0.008),
    "HIGH": build_scenario(future_years, co2_rate=4.5,  ch4_rate=3.0,  n2o_rate=0.6,
                            aerosol_trend=-0.002, urbanization_rate=0.012),
}

fig, ax = plt.subplots(figsize=(13, 6))

# Plot historical
ax.plot(df["year"], df[TARGET], color="white", linewidth=1.5,
        alpha=0.7, label="Historical (1880–2024)", zorder=3)

scenario_preds = {}
for sc_name, sc_df in scenarios.items():
    sc_scaled = scaler.transform(sc_df[FEATURES])
    preds = best_model.predict(sc_scaled)
    scenario_preds[sc_name] = preds
    ax.plot(future_years, preds, color=SCENARIO_COLORS[sc_name],
            linewidth=2.2, label=f"{sc_name} scenario", zorder=4)
    ax.fill_between(future_years, preds, alpha=0.12,
                    color=SCENARIO_COLORS[sc_name])
    ax.annotate(f"{preds[-1]:.2f}°C",
                xy=(2050, preds[-1]),
                xytext=(2051, preds[-1]),
                fontsize=9, color=SCENARIO_COLORS[sc_name],
                fontweight="bold")

ax.axhline(1.5, color="#f39c12", linewidth=1, linestyle="--",
           alpha=0.8, label="Paris +1.5 °C")
ax.axhline(2.0, color="#e74c3c", linewidth=1, linestyle="--",
           alpha=0.8, label="Paris +2.0 °C")
ax.axvline(2024, color="white", linewidth=1, linestyle=":", alpha=0.5)
ax.text(2024.3, ax.get_ylim()[0] + 0.1, "Forecast →", fontsize=9,
        color="white", alpha=0.6)

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Temperature Anomaly (°C)", fontsize=11)
ax.set_title(f"Global Temperature Forecast to 2050 — Three Emission Scenarios\n"
             f"Model: {best_name}  |  LOW = aggressive mitigation  |  MED = BAU  |  HIGH = accelerated emissions",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(1880, 2053)
plt.tight_layout()
plt.savefig("outputs/forecast_2050.png", dpi=150, bbox_inches="tight")
plt.close()
print("🖼️   Saved → outputs/forecast_2050.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10. FIGURE 6 — Residual Analysis (Best Model)
# ══════════════════════════════════════════════════════════════════════════════
y_pred_best = results[best_name]["y_pred"]
residuals = y_test.values - y_pred_best

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Residual Analysis — {best_name}", fontsize=13, fontweight="bold")

axes[0].scatter(y_pred_best, residuals, color=PALETTE[best_name],
                alpha=0.7, edgecolors="white", linewidths=0.4, s=60)
axes[0].axhline(0, color="white", linewidth=1.2, linestyle="--")
axes[0].set_xlabel("Predicted Temperature Anomaly (°C)", fontsize=11)
axes[0].set_ylabel("Residual (Actual − Predicted)", fontsize=11)
axes[0].set_title("Residuals vs Predicted")

axes[1].hist(residuals, bins=15, color=PALETTE[best_name],
             edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="white", linewidth=1.5, linestyle="--")
axes[1].set_xlabel("Residual (°C)", fontsize=11)
axes[1].set_ylabel("Count", fontsize=11)
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("outputs/residual_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("🖼️   Saved → outputs/residual_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# 11. EXPORT FORECAST DATA
# ══════════════════════════════════════════════════════════════════════════════
forecast_df = pd.DataFrame({"year": future_years})
for sc_name, preds in scenario_preds.items():
    forecast_df[f"temp_anomaly_{sc_name}"] = preds
forecast_df.to_csv("outputs/forecast_2050_data.csv", index=False)
print("📄  Forecast data saved → outputs/forecast_2050_data.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 12. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PIPELINE COMPLETE")
print("=" * 65)
print(f"\n  Best Model  : {best_name}")
print(f"  R²          : {results[best_name]['R2']:.4f}")
print(f"  RMSE        : {results[best_name]['RMSE']:.4f} °C")
print(f"  MAE         : {results[best_name]['MAE']:.4f} °C")
print(f"  CV R²       : {results[best_name]['CV_mean']:.4f} ± {results[best_name]['CV_std']:.4f}")
print(f"\n  2050 Projections ({best_name}):")
for sc_name, preds in scenario_preds.items():
    print(f"    {sc_name:<5} →  {preds[-1]:.3f} °C")
print("\n  Outputs saved to outputs/ and models/")
print("=" * 65)

