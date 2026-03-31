# 🌡️ Predicting Global Temperature Rise Using ML Models

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/Nelvinebi/Predicting-Global-Temperature-Rise-Using-ML-Models?style=for-the-badge&color=yellow)

> A machine learning pipeline that simulates **145 years of climate history (1880–2024)** and forecasts **global mean temperature anomalies to 2050** under three emission scenarios powered by synthetic climate-forcing data and ensemble ML models.

<div align="center">

[![Live Dashboard](https://img.shields.io/badge/🚀%20Click%20for%20Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://predicting-global-temperature-rise-using-ml-models-jowqgmqy2ug.streamlit.app/)

</div>

---

## 📌 Problem

Global temperature rise is one of the most consequential challenges of our time, yet understanding the **quantitative relationship between climate-forcing factors and warming trends** remains computationally intensive and data-restricted for many researchers. Access to clean, structured, analysis-ready climate datasets is often a barrier for data scientists entering the climate space.

There is a clear need for accessible, reproducible ML pipelines that can model the influence of greenhouse gases, aerosols, solar variability, and ENSO patterns on temperature anomalies — and project those trends into future emission scenarios.

---

## 🎯 Objective

- Simulate **synthetic climate-forcing data** spanning 1880–2024 (144 years)
- Train and compare multiple ML regression models on this dataset
- Identify the **most influential drivers** of global temperature rise using permutation feature importance
- Forecast global mean temperature anomalies to **2050** under three distinct emission scenarios:
  - 🟢 **LOW** : aggressive mitigation / net-zero pathway
  - 🟡 **MED** : business-as-usual / moderate action
  - 🔴 **HIGH** : accelerated emissions / no policy change
- Provide an **educational and reproducible framework** for climate data science

---

## 🗂️ Dataset

The dataset is synthetically generated to mirror realistic climate-forcing dynamics, covering 1880–2024.

| Feature | Description | Unit |
|---------|-------------|------|
| `year` | Calendar year | — |
| `co2_ppm` | Atmospheric CO₂ concentration | ppm |
| `ch4_ppb` | Atmospheric methane (CH₄) concentration | ppb |
| `n2o_ppb` | Nitrous oxide (N₂O) concentration | ppb |
| `aerosol_optical_depth` | Aerosol optical depth — cooling effect | — |
| `solar_irradiance_anom` | Total solar irradiance anomaly | W/m² |
| `enso_index` | El Niño–Southern Oscillation index | — |
| `volcanic_forcing` | Volcanic radiative forcing | W/m² |
| `land_use_index` | Land use change intensity index | 0–1 |
| `urbanization_index` | Urban expansion index | 0–1 |
| `temp_anomaly_C` | **Target:** Global mean temperature anomaly | °C |

- **Size:** 145 samples (1 per year, 1880–2024)
- **Format:** Excel (`global_temp_synthetic.xlsx`) + CSV fallback
- **Features:** 9 climate-forcing predictors → 1 regression target
- **No missing values** across all 145 rows × 11 columns
- **Generation:** Synthetic, calibrated to replicate observed 20th–21st century warming trends

---

## 🛠️ Tools & Technologies

- **Language:** Python 3.x
- **ML Models:** Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor
- **Core Libraries:** scikit-learn, NumPy, Pandas, openpyxl, joblib
- **Visualisation:** Matplotlib, Seaborn
- **Feature Importance:** Permutation Importance on held-out test set (scikit-learn)
- **Cross-Validation:** 5-Fold KFold (shuffle=True) for robust generalisation estimates
- **Model Persistence:** joblib — best model saved to `models/best_model.pkl`
- **Scenario Modelling:** Custom projection functions for LOW / MED / HIGH pathways

---

## ⚙️ Methodology / Project Workflow

1. **Data Loading & Inspection** — Load 145-year Excel dataset; verify all 9 features and target; confirm zero null values
2. **Exploratory Data Analysis (EDA)** : Generate correlation heatmap across all 10 variables; plot historical temperature anomaly trend with 10-year rolling mean and Paris Agreement thresholds
3. **Feature Engineering** : Use all 9 climate-forcing variables as predictors; apply `StandardScaler` for fair cross-model comparison
4. **Train / Test Split** : 80/20 random shuffle split (116 train / 29 test, `random_state=42`)
5. **Model Training** : Train three regressors: Ridge Regression, Random Forest (n=300, max_depth=8), Gradient Boosting (n=300, lr=0.05, subsample=0.8)
6. **Model Evaluation** : Compare MAE, RMSE, R² on test set + 5-Fold KFold cross-validation R² for each model
7. **Permutation Feature Importance** : 30-repeat permutation test on held-out test set using best model; rank all 9 forcing variables
8. **Scenario Forecasting** : Project temperature anomalies 2025–2050 under LOW / MED / HIGH emission pathways; annotate 2050 endpoints and Paris Agreement thresholds
9. **Residual Analysis** : Residuals vs predicted scatter + residual distribution histogram for best model
10. **Export All Outputs** : 6 publication-ready plots, model metrics text file, forecast CSV, and saved model (joblib)

---

## 📊 Key Features

- ✅ **145-year synthetic climate dataset** (1880–2024) with **9 climate-forcing variables** including volcanic forcing, land use, and urbanisation indices
- ✅ **Three ML models** trained, evaluated, and benchmarked side-by-side (Ridge, Random Forest, Gradient Boosting) : all achieving R² > 0.97
- ✅ **5-Fold cross-validated performance** : CV R² reported alongside test metrics for every model
- ✅ **Permutation feature importance** (30 repeats on test set) : ranks all 9 GHG/forcing drivers of warming
- ✅ **Multi-scenario forecasting to 2050** : LOW / MED / HIGH emission pathways with 2050 endpoint annotations
- ✅ **6 publication-ready visualisations** : correlation heatmap, historical trend, model comparison, feature importance, forecast, residual analysis
- ✅ **Best model saved** via joblib (`models/best_model.pkl`) : ready for reuse or deployment
- ✅ **Reproducible single-script pipeline** : all outputs regenerated from one `python` command

---

## 📸 Visualisations

### 🔹 Correlation Heatmap — Climate Forcing Variables
> Pearson correlation matrix across all 9 predictors and the temperature target reveals CO₂, CH₄, and N₂O as the most strongly correlated with warming

![Correlation Heatmap](outputs/https://github.com/Nelvinebi/Predicting-Global-Temperature-Rise-Using-ML-Models/blob/b9787cabb3c53042877a3c2cd41d2d5d679b7ffc/Outputs/correlation_heatmap.png)

---

### 🔹 Historical Temperature Anomaly Trend (1880–2024)
> 144-year warming trajectory with 10-year rolling mean and Paris Agreement +1.5°C / +2.0°C thresholds overlaid

![Temperature Trend](outputs/https://github.com/Nelvinebi/Predicting-Global-Temperature-Rise-Using-ML-Models/blob/b9787cabb3c53042877a3c2cd41d2d5d679b7ffc/Outputs/temperature_trend.png)

---

### 🔹 Model Comparison — Actual vs Predicted (Test Set)
> Side-by-side comparison of all three models against held-out test data

![Model Comparison](outputs/https://github.com/Nelvinebi/Predicting-Global-Temperature-Rise-Using-ML-Models/blob/b9787cabb3c53042877a3c2cd41d2d5d679b7ffc/Outputs/model_comparison.png)

---

### 🔹 Permutation Feature Importance
> Which climate-forcing variables matter most? Ranked by mean permutation importance on the test set (30 repeats)

![Feature Importance](outputs/https://github.com/Nelvinebi/Predicting-Global-Temperature-Rise-Using-ML-Models/blob/b9787cabb3c53042877a3c2cd41d2d5d679b7ffc/Outputs/feature_importance.pngg)

---

### 🔹 Temperature Forecast to 2050 — Three Emission Scenarios
> Projected warming under LOW (aggressive mitigation), MED (business-as-usual), and HIGH (accelerated emissions)

![Forecast 2050](outputs/https://github.com/Nelvinebi/Predicting-Global-Temperature-Rise-Using-ML-Models/blob/b9787cabb3c53042877a3c2cd41d2d5d679b7ffc/Outputs/forecast_2050.png)

---

### 🔹 Residual Analysis — Best Model
> Residuals vs predicted values + residual distribution to check model assumptions

![Residual Analysis](outputs/https://github.com/Nelvinebi/Predicting-Global-Temperature-Rise-Using-ML-Models/blob/b9787cabb3c53042877a3c2cd41d2d5d679b7ffc/Outputs/residual_analysis.png)

> 📌 *All plots are saved at 150 dpi in the `/outputs/` folder.*

---

## 📈 Results & Insights

### Model Performance Comparison

| Model | MAE (°C) | RMSE (°C) | R² Score | CV R² (5-fold) |
|-------|----------|-----------|----------|----------------|
| Ridge Regression | 0.0579 | 0.0756 | **0.9934** | 0.9894 ± 0.0025 |
| Gradient Boosting | 0.0729 | 0.1177 | 0.9839 | 0.9848 ± 0.0048 |
| Random Forest | 0.0860 | 0.1388 | 0.9776 | 0.9796 ± 0.0057 |

> 🏆 **Ridge Regression achieved the best performance** (R² = 0.9934), confirming that the dominant relationships between climate-forcing variables and temperature anomalies are strongly linear — consistent with established climate physics.

### Key Insights

- 🔍 **CO₂ is the single strongest driver** of temperature anomalies — consistent with IPCC AR6 findings on greenhouse gas dominance
- 🔍 **Aerosol optical depth exerts a measurable cooling effect**, partially masking GHG-driven warming, especially pre-1970
- 🔍 **Volcanic forcing** introduces short-term cooling pulses visible in the residual analysis
- 🔍 **All three models achieve R² > 0.97**, confirming the strong physical signal in this synthetic dataset
- 🔍 **Under the HIGH scenario**, the model projects a temperature anomaly of **+3.94°C by 2050** — far exceeding the Paris Agreement's +2°C threshold
- 🔍 **Under the LOW scenario**, warming is projected at **+2.99°C** — still above 1.5°C, highlighting the urgency of early and aggressive mitigation
- 🔍 **Ridge Regression (R²=0.9934) outperforms tree models**, confirming that the temperature–GHG relationship in this dataset is largely linear and well-captured by a regularised linear model

---

## 🚀 Live Demo / Notebook Viewer

📓 **[View the Notebook on nbviewer →](https://predicting-global-temperature-rise-using-ml-models-jowqgmqy2ug.streamlit.app/)**

> *No Streamlit app deployed for this project. Run locally using the instructions below.*

---

## 📁 Repository Structure

```
📦 Predicting-Global-Temperature-Rise-Using-ML-Models/
├── 📂 Data/
│   └── global_temp_synthetic.xlsx        # 145-year synthetic climate dataset (11 columns)
├── 📂 outputs/
│   ├── correlation_heatmap.png           # Feature correlation matrix
│   ├── temperature_trend.png             # Historical anomaly trend 1880–2024
│   ├── model_comparison.png              # Actual vs predicted — all 3 models
│   ├── feature_importance.png            # Permutation importance ranking
│   ├── forecast_2050.png                 # LOW / MED / HIGH scenario projections
│   ├── residual_analysis.png             # Residuals vs predicted + distribution
│   ├── forecast_2050_data.csv            # Numeric forecast values (exportable)
│   └── model_metrics.txt                 # MAE, RMSE, R², CV R² for all models
├── 📂 models/
│   └── best_model.pkl                    # Saved best model + scaler (joblib)
├── predict_global_temperature_ml.py      # Full ML pipeline (single-script)
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Nelvinebi/Predicting-Global-Temperature-Rise-Using-ML-Models.git
cd Predicting-Global-Temperature-Rise-Using-ML-Models

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full ML pipeline (generates all outputs automatically)
python predict_global_temperature_ml.py
```

**What the script produces automatically:**

| Output | Location |
|--------|----------|
| Correlation heatmap | `outputs/correlation_heatmap.png` |
| Historical trend plot | `outputs/temperature_trend.png` |
| Model comparison chart | `outputs/model_comparison.png` |
| Feature importance plot | `outputs/feature_importance.png` |
| 2050 forecast chart | `outputs/forecast_2050.png` |
| Residual analysis | `outputs/residual_analysis.png` |
| Forecast CSV | `outputs/forecast_2050_data.csv` |
| Model metrics summary | `outputs/model_metrics.txt` |
| Saved best model | `models/best_model.pkl` |

### Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
openpyxl
joblib
```

---

## ⚠️ Limitations & Future Work

**Current Limitations:**
- Dataset is **synthetic** — results are illustrative and not a substitute for real observational records (e.g., NASA GISS, HadCRUT5, Berkeley Earth)
- **Only 145 data points** — limits model complexity and statistical power; tree models benefit less from this small sample
- Models produce a **single global mean** — no spatial (regional/gridded) resolution
- **Scenario projections use simplified linear forcing assumptions** — not full Earth System Model dynamics (e.g., CMIP6)
- No **prediction intervals or uncertainty bounds** on 2050 forecasts

**Future Improvements:**
- 🔁 Replace synthetic data with **real NASA GISS / NOAA / Berkeley Earth observational records**
- 🌐 Extend to **regional temperature modelling** using gridded reanalysis data (ERA5, CMIP6 ensembles)
- 📉 Add **bootstrap confidence intervals** and ensemble spread to scenario forecasts
- 🤖 Integrate **LSTM / Transformer time-series models** to capture temporal dependencies
- 🖥️ Deploy as a **Streamlit interactive dashboard** for real-time scenario exploration
- 🧪 Apply **SHAP values** for deeper, interaction-aware feature attribution

---

## 👤 Author

**Name:** Agbozu Ebingiye Nelvin

🌍 Environmental Data Scientist | GIS & Remote Sensing | Climate ML
📍 Port Harcourt, Nigeria

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/agbozu-ebi/)
[![GitHub](https://img.shields.io/badge/GitHub-Nelvinebi-181717?style=flat-square&logo=github)](https://github.com/Nelvinebi)
[![Email](https://img.shields.io/badge/Email-nelvinebingiye%40gmail.com-D14836?style=flat-square&logo=gmail)](mailto:nelvinebingiye@gmail.com)
[![Streamlit Apps](https://img.shields.io/badge/Streamlit%20Apps-FF4B4B?style=flat-square&logo=streamlit)](https://share.streamlit.io/user/nelvinebi)

---

## 📄 License

This project is licensed under the **MIT License** free to use, adapt, and build upon for research and education.
See the [LICENSE](LICENSE) file for full details.

---

## 🙌 Acknowledgements

- Climate forcing conceptual framework inspired by **IPCC AR6 Working Group I** findings
- Synthetic data generation approach informed by **NASA GISS Surface Temperature Analysis (GISTEMP)**
- ML methodology follows best practices from **scikit-learn** documentation

---

<div align="center">

⭐ **If this project helped you, please consider starring the repo!**

*Part of a broader portfolio of Environmental Data Science projects focused on the Niger Delta and global climate systems.*

🔗 [View All Projects](https://github.com/Nelvinebi?tab=repositories) · [Connect on LinkedIn](https://www.linkedin.com/in/agbozu-ebi/) · [Live Apps](https://share.streamlit.io/user/nelvinebi)

</div>
