# 🌍 Predicting Global Temperature Rise Using ML Models

This project demonstrates how **synthetic climate-forcing data** and **machine learning models** can be used to predict and forecast **global mean temperature anomalies**. It generates synthetic greenhouse gas, aerosol, solar, and ENSO datasets, then applies ML models to simulate historical warming and forecast future climate scenarios.

---

## 📌 Features
- Generates **synthetic dataset** (1880–2024, >100 years).
- Synthesizes **global mean temperature anomaly** (°C).
- Trains multiple ML models:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluates models (MAE, RMSE, R²).
- Estimates **feature importance** (permutation method).
- Forecasts to **2050** under three scenarios:
  - **LOW**: aggressive mitigation
  - **MED**: medium/BAU
  - **HIGH**: high emissions
- Exports dataset, metrics, and visualizations.

---

## 📂 Project Structure
├── predict_global_temperature_ml.py # Main script
├── data/
│ └── global_temp_synthetic.csv # Generated dataset
├── outputs/
│ ├── model_metrics.txt # Metrics summary
│ ├── feature_importance.png # Feature importance plot
│ └── forecast_2050.png # Forecast visualization
└── README.md

yaml
Copy code

---

## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/Predicting-Global-Temperature-Rise-Using-ML-Models.git
cd Predicting-Global-Temperature-Rise-Using-ML-Models

pip install -r requirements.txt
🚀 Usage
Run the script:

bash
Copy code
python predict_global_temperature_ml.py
This will generate the dataset and save metrics/plots under the data/ and outputs/ folders.

📊 Example Outputs
feature_importance.png → shows key drivers of temperature rise.

forecast_2050.png → compares synthetic historical temps with LOW/MED/HIGH scenarios to 2050.

model_metrics.txt → contains MAE, RMSE, R² for each ML model.

🌍 Applications
Educational demonstration of climate-forcing interactions.

Testing ML workflows on synthetic environmental datasets.

Climate change communication and scenario analysis.


## Author
Name: Agbozu Ebingiye Nelvin

Github: https://github.com/Nelvinebi

LinkedIn: *https://www.linkedin.com/in/agbozu-ebi/


📜 License
MIT License — free to use and adapt for research and education.

yaml
Copy code

---

## 📦 requirements.txt

```txt
numpy
pandas
matplotlib
scikit-learn
