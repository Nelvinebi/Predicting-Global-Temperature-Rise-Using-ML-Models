"""
================================================================================
  Global Temperature Rise — ML Dashboard
  Author : Agbozu Ebingiye Nelvin
  GitHub : https://github.com/Nelvinebi
================================================================================
  Run with:  streamlit run dashboard.py
  Place this file next to your Data/ folder (which contains
  global_temp_synthetic.xlsx) or adjust DATA_PATH below.
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

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

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Temperature ML Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark science-lab theme */
.stApp {
    background: #0a0e14;
    color: #c9d1d9;
}

/* Hero header */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #112240 50%, #0d1117 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(231,76,60,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(52,152,219,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e74c3c, #f39c12, #3498db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
}
.hero-sub {
    color: #8b949e;
    font-size: 1rem;
    font-weight: 400;
    font-family: 'DM Mono', monospace;
    margin: 0;
}

/* Metric cards */
.metric-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #e74c3c44; }
.metric-label {
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 0;
}
.metric-delta {
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    margin-top: 0.2rem;
}
.delta-good { color: #3fb950; }
.delta-warn { color: #d29922; }
.delta-bad  { color: #f85149; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e6edf3;
    border-left: 3px solid #e74c3c;
    padding-left: 0.8rem;
    margin: 2rem 0 1rem 0;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label {
    color: #8b949e;
    font-size: 0.8rem;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #21262d;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.7rem 1.4rem;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #e6edf3 !important;
    background: #161b22 !important;
    border-bottom: 2px solid #e74c3c !important;
}

/* Dataframe */
.stDataFrame { border: 1px solid #21262d; border-radius: 8px; }

/* Expander */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Footer tag */
.footer-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #484f58;
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #21262d;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    "co2_ppm", "ch4_ppb", "n2o_ppb",
    "aerosol_optical_depth", "solar_irradiance_anom", "enso_index",
    "volcanic_forcing", "land_use_index", "urbanization_index",
]
FEATURE_LABELS = {
    "co2_ppm":               "CO₂ (ppm)",
    "ch4_ppb":               "CH₄ (ppb)",
    "n2o_ppb":               "N₂O (ppb)",
    "aerosol_optical_depth": "Aerosol Opt. Depth",
    "solar_irradiance_anom": "Solar Irradiance Anom.",
    "enso_index":            "ENSO Index",
    "volcanic_forcing":      "Volcanic Forcing",
    "land_use_index":        "Land Use Index",
    "urbanization_index":    "Urbanisation Index",
}
TARGET = "temp_anomaly_C"
PALETTE = {
    "Ridge":             "#4CAF50",
    "Random Forest":     "#2196F3",
    "Gradient Boosting": "#FF5722",
}
SCENARIO_COLORS = {"LOW": "#27ae60", "MED": "#f39c12", "HIGH": "#e74c3c"}

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#21262d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#21262d",
})

# ── Data path ─────────────────────────────────────────────────────────────────
DATA_PATH = "Data/global_temp_synthetic.xlsx"   # adjust if needed

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading climate dataset …")
def load_data(path):
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        try:
            df = pd.read_excel(path.replace(".xlsx", ".csv").replace("Data/", "Data/"))
        except Exception:
            df = pd.read_csv(path.replace(".xlsx", ".csv"))
    return df

# ── Train & cache models ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training ML models …")
def train_models(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models_def = {
        "Ridge":             Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(n_estimators=300, max_depth=8,
                                                    random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                        max_depth=4, random_state=42,
                                                        subsample=0.8),
    }
    results = {}
    for name, model in models_def.items():
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

    best_name = max(results, key=lambda n: results[n]["R2"])
    return results, scaler, X_train, X_test, y_train, y_test, best_name

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌡️ Dashboard Controls")
    st.markdown("---")

    data_path = st.text_input("Data path", value=DATA_PATH,
                               help="Path to global_temp_synthetic.xlsx")
    st.markdown("---")

    st.markdown("**Year Range Filter**")
    year_range = st.slider("", 1880, 2024, (1880, 2024), step=1, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Forecast Settings**")
    forecast_horizon = st.slider("Forecast to year", 2030, 2100, 2050, step=5)
    show_paris = st.checkbox("Show Paris thresholds", value=True)
    show_rolling = st.checkbox("Show 10-yr rolling mean", value=True)

    st.markdown("---")
    st.markdown("**Feature to Explore**")
    selected_feature = st.selectbox(
        "", list(FEATURE_LABELS.keys()),
        format_func=lambda x: FEATURE_LABELS[x],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:0.7rem;color:#484f58'>"
        "Agbozu Ebingiye Nelvin<br>github.com/Nelvinebi</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
try:
    df_full = load_data(data_path)
except Exception as e:
    st.error(f"❌  Could not load data from `{data_path}`. Error: {e}")
    st.stop()

df = df_full[(df_full["year"] >= year_range[0]) & (df_full["year"] <= year_range[1])].copy()

# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🌍 Global Temperature Rise</p>
  <p class="hero-sub">Machine Learning Analysis · 1880–2024 Synthetic Climate Dataset · Three Emission Scenarios</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
latest_temp  = df_full[TARGET].iloc[-1]
baseline_temp = df_full[TARGET].iloc[:20].mean()
delta_temp   = latest_temp - baseline_temp
pct_rise     = df_full[TARGET].max() - df_full[TARGET].min()
year_span    = int(df_full["year"].max()) - int(df_full["year"].min())
latest_co2   = df_full["co2_ppm"].iloc[-1]

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Latest Temp. Anomaly</div>
      <div class="metric-value">{latest_temp:+.3f} °C</div>
      <div class="metric-delta delta-bad">vs pre-industrial baseline</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Rise Since 1880</div>
      <div class="metric-value">{delta_temp:+.3f} °C</div>
      <div class="metric-delta {'delta-bad' if delta_temp > 1.5 else 'delta-warn'}">
        {'⚠️ Exceeds Paris +1.5°C' if delta_temp > 1.5 else 'Approaching Paris limit'}
      </div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Latest CO₂ Level</div>
      <div class="metric-value">{latest_co2:.1f} ppm</div>
      <div class="metric-delta delta-warn">Pre-industrial: ~280 ppm</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Dataset Span</div>
      <div class="metric-value">{year_span} yrs</div>
      <div class="metric-delta delta-good">{len(df_full)} annual observations</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
results, scaler, X_train, X_test, y_train, y_test, best_name = train_models(df_full)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 Historical Trend",
    "🤖 Model Performance",
    "🔬 Feature Analysis",
    "🔮 2050 Forecast",
    "🗃️ Raw Data",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — HISTORICAL TREND
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">Global Mean Temperature Anomaly</div>', unsafe_allow_html=True)

    col_main, col_side = st.columns([3, 1])

    with col_main:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(df["year"], df[TARGET], alpha=0.18, color="#e74c3c")
        ax.plot(df["year"], df[TARGET], color="#e74c3c", linewidth=1.6,
                label="Temp. Anomaly (°C)", zorder=3)
        ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--", alpha=0.5)

        if show_paris:
            ax.axhline(1.5, color="#f39c12", linewidth=1.2, linestyle="--",
                       alpha=0.85, label="Paris +1.5 °C")
            ax.axhline(2.0, color="#e74c3c", linewidth=1.2, linestyle="--",
                       alpha=0.85, label="Paris +2.0 °C")

        if show_rolling:
            rolling = df.set_index("year")[TARGET].rolling(10, center=True).mean()
            ax.plot(df["year"], rolling.values, color="#f0f6fc", linewidth=2.2,
                    linestyle="-", label="10-yr rolling mean", zorder=4)

        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Temperature Anomaly (°C)", fontsize=10)
        ax.set_title(f"Global Mean Temperature Anomaly ({year_range[0]}–{year_range[1]})",
                     fontsize=12, fontweight="bold", pad=10)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_side:
        st.markdown("**Period Stats**")
        filtered_mean = df[TARGET].mean()
        filtered_max  = df[TARGET].max()
        filtered_min  = df[TARGET].min()
        st.metric("Mean Anomaly",  f"{filtered_mean:+.3f} °C")
        st.metric("Max Anomaly",   f"{filtered_max:+.3f} °C")
        st.metric("Min Anomaly",   f"{filtered_min:+.3f} °C")
        st.metric("Std Deviation", f"{df[TARGET].std():.3f} °C")

        warmest = df.loc[df[TARGET].idxmax(), "year"]
        coolest = df.loc[df[TARGET].idxmin(), "year"]
        st.metric("Warmest Year", str(int(warmest)))
        st.metric("Coolest Year", str(int(coolest)))

    # Feature over time
    st.markdown(f'<div class="section-header">Feature Trend: {FEATURE_LABELS[selected_feature]}</div>',
                unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(12, 3.5))
    ax2.plot(df["year"], df[selected_feature],
             color="#3498db", linewidth=1.5, label=FEATURE_LABELS[selected_feature])
    ax2.fill_between(df["year"], df[selected_feature], alpha=0.12, color="#3498db")
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel(FEATURE_LABELS[selected_feature], fontsize=10)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">Model Evaluation Summary</div>', unsafe_allow_html=True)

    # Metrics table
    metrics_data = []
    for name, r in results.items():
        metrics_data.append({
            "Model": name,
            "MAE":   round(r["MAE"],  4),
            "RMSE":  round(r["RMSE"], 4),
            "R²":    round(r["R2"],   4),
            "CV R² (mean)": round(r["CV_mean"], 4),
            "CV R² (std)":  round(r["CV_std"],  4),
            "Best": "🏆" if name == best_name else "",
        })
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(
        df_metrics.style
            .highlight_max(subset=["R²", "CV R² (mean)"], color="#1a3a2a")
            .highlight_min(subset=["MAE", "RMSE"], color="#1a3a2a")
            .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}",
                     "CV R² (mean)": "{:.4f}", "CV R² (std)": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown('<div class="section-header">Actual vs Predicted — Test Set</div>',
                unsafe_allow_html=True)

    test_years = df_full["year"].iloc[X_test.index].values
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig3.patch.set_facecolor("#0d1117")

    for ax, (name, r) in zip(axes3, results.items()):
        ax.plot(test_years, y_test.values, color="#f0f6fc", linewidth=1.8,
                label="Actual", zorder=3)
        ax.plot(test_years, r["y_pred"], color=PALETTE[name], linewidth=1.8,
                linestyle="--", label="Predicted", zorder=4)
        ax.fill_between(test_years, y_test.values, r["y_pred"],
                        alpha=0.15, color=PALETTE[name])
        ax.set_title(f"{name}\nR²={r['R2']:.4f}  RMSE={r['RMSE']:.4f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Year", fontsize=9)
        ax.legend(fontsize=8)
        if ax == axes3[0]:
            ax.set_ylabel("Temp. Anomaly (°C)", fontsize=9)

    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # Residuals
    st.markdown(f'<div class="section-header">Residual Analysis — {best_name}</div>',
                unsafe_allow_html=True)
    y_pred_best = results[best_name]["y_pred"]
    residuals   = y_test.values - y_pred_best

    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 4.5))
    fig4.patch.set_facecolor("#0d1117")

    axes4[0].scatter(y_pred_best, residuals, color=PALETTE[best_name],
                     alpha=0.7, edgecolors="#f0f6fc", linewidths=0.4, s=55)
    axes4[0].axhline(0, color="#f0f6fc", linewidth=1.2, linestyle="--")
    axes4[0].set_xlabel("Predicted (°C)", fontsize=10)
    axes4[0].set_ylabel("Residual (°C)", fontsize=10)
    axes4[0].set_title("Residuals vs Predicted", fontsize=11)

    axes4[1].hist(residuals, bins=16, color=PALETTE[best_name],
                  edgecolor="#0d1117", alpha=0.85)
    axes4[1].axvline(0, color="#f0f6fc", linewidth=1.5, linestyle="--")
    axes4[1].set_xlabel("Residual (°C)", fontsize=10)
    axes4[1].set_ylabel("Count", fontsize=10)
    axes4[1].set_title("Residual Distribution", fontsize=11)

    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        fig5.patch.set_facecolor("#0d1117")
        corr = df_full[FEATURES + [TARGET]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    linewidths=0.4, ax=ax5, vmin=-1, vmax=1,
                    annot_kws={"size": 7.5},
                    xticklabels=[FEATURE_LABELS.get(c, c) for c in corr.columns],
                    yticklabels=[FEATURE_LABELS.get(c, c) for c in corr.index])
        ax5.set_title("Feature Correlation Matrix", fontsize=11, fontweight="bold")
        ax5.tick_params(labelsize=7.5)
        fig5.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

    with col_b:
        st.markdown('<div class="section-header">Permutation Feature Importance</div>',
                    unsafe_allow_html=True)

        best_model = results[best_name]["model"]
        X_test_sc  = scaler.transform(X_test)

        @st.cache_data(show_spinner="Computing permutation importance …")
        def get_perm_importance(_model, _X_test_sc, _y_test):
            perm = permutation_importance(_model, _X_test_sc, _y_test,
                                          n_repeats=20, random_state=42, n_jobs=-1)
            return perm.importances_mean, perm.importances_std

        imp_mean, imp_std = get_perm_importance(best_model, X_test_sc, y_test)

        feat_imp = pd.DataFrame({
            "Feature":    FEATURES,
            "Label":      [FEATURE_LABELS[f] for f in FEATURES],
            "Importance": imp_mean,
            "Std":        imp_std,
        }).sort_values("Importance", ascending=True)

        fig6, ax6 = plt.subplots(figsize=(8, 6))
        fig6.patch.set_facecolor("#0d1117")
        colors_fi = ["#e74c3c" if v > 0 else "#484f58" for v in feat_imp["Importance"]]
        bars = ax6.barh(feat_imp["Label"], feat_imp["Importance"],
                        xerr=feat_imp["Std"], color=colors_fi, capsize=4,
                        edgecolor="#0d1117", linewidth=0.5)
        ax6.axvline(0, color="#8b949e", linewidth=0.8)
        ax6.set_xlabel("Mean Permutation Importance (Δ R²)", fontsize=9)
        ax6.set_title(f"Feature Importance — {best_name}", fontsize=11, fontweight="bold")
        ax6.tick_params(labelsize=8.5)
        for bar, val in zip(bars, feat_imp["Importance"]):
            ax6.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=8)
        fig6.tight_layout()
        st.pyplot(fig6, use_container_width=True)
        plt.close(fig6)

    # Scatter: selected feature vs target
    st.markdown(f'<div class="section-header">{FEATURE_LABELS[selected_feature]} vs Temperature Anomaly</div>',
                unsafe_allow_html=True)
    fig7, ax7 = plt.subplots(figsize=(10, 4))
    fig7.patch.set_facecolor("#0d1117")
    sc = ax7.scatter(df_full[selected_feature], df_full[TARGET],
                     c=df_full["year"], cmap="plasma", alpha=0.75,
                     edgecolors="none", s=35)
    cbar = plt.colorbar(sc, ax=ax7)
    cbar.set_label("Year", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax7.set_xlabel(FEATURE_LABELS[selected_feature], fontsize=10)
    ax7.set_ylabel("Temp. Anomaly (°C)", fontsize=10)
    r_val = df_full[[selected_feature, TARGET]].corr().iloc[0, 1]
    ax7.set_title(f"Pearson r = {r_val:.3f}", fontsize=11, fontweight="bold")
    fig7.tight_layout()
    st.pyplot(fig7, use_container_width=True)
    plt.close(fig7)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">2050 Scenario Forecast</div>', unsafe_allow_html=True)

    future_years = np.arange(2025, forecast_horizon + 1)
    base = df_full[df_full["year"] == df_full["year"].max()][FEATURES].iloc[0].to_dict()

    def build_scenario(years, co2_rate, ch4_rate, aerosol_trend, urbanization_rate):
        rows = []
        for i, yr in enumerate(years):
            t = i + 1
            rows.append({
                "co2_ppm":               base["co2_ppm"]              + co2_rate * t,
                "ch4_ppb":               base["ch4_ppb"]              + ch4_rate * t,
                "n2o_ppb":               base["n2o_ppb"]              + 0.3 * t,
                "aerosol_optical_depth": base["aerosol_optical_depth"] + aerosol_trend * t,
                "solar_irradiance_anom": base["solar_irradiance_anom"] + np.random.normal(0, 0.02),
                "enso_index":            np.random.normal(0, 0.5),
                "volcanic_forcing":      np.random.normal(-0.05, 0.05),
                "land_use_index":        min(base["land_use_index"]    + 0.005 * t, 1.0),
                "urbanization_index":    min(base["urbanization_index"]+ urbanization_rate * t, 1.0),
            })
        return pd.DataFrame(rows)

    np.random.seed(42)
    scenarios = {
        "LOW":  build_scenario(future_years, co2_rate=0.5,  ch4_rate=0.3,
                               aerosol_trend=0.002,  urbanization_rate=0.004),
        "MED":  build_scenario(future_years, co2_rate=2.5,  ch4_rate=1.5,
                               aerosol_trend=0.0,    urbanization_rate=0.008),
        "HIGH": build_scenario(future_years, co2_rate=4.5,  ch4_rate=3.0,
                               aerosol_trend=-0.002, urbanization_rate=0.012),
    }

    scenario_preds = {}
    for sc_name, sc_df in scenarios.items():
        sc_scaled = scaler.transform(sc_df[FEATURES])
        scenario_preds[sc_name] = results[best_name]["model"].predict(sc_scaled)

    fig8, ax8 = plt.subplots(figsize=(13, 6))
    fig8.patch.set_facecolor("#0d1117")

    ax8.plot(df_full["year"], df_full[TARGET], color="#f0f6fc",
             linewidth=1.4, alpha=0.7, label="Historical (1880–2024)", zorder=3)

    for sc_name, preds in scenario_preds.items():
        ax8.plot(future_years, preds, color=SCENARIO_COLORS[sc_name],
                 linewidth=2.2, label=f"{sc_name} scenario", zorder=4)
        ax8.fill_between(future_years, preds, alpha=0.1,
                         color=SCENARIO_COLORS[sc_name])
        ax8.annotate(f"{preds[-1]:.2f}°C",
                     xy=(future_years[-1], preds[-1]),
                     xytext=(future_years[-1] + 0.8, preds[-1]),
                     fontsize=9, color=SCENARIO_COLORS[sc_name], fontweight="bold")

    if show_paris:
        ax8.axhline(1.5, color="#f39c12", linewidth=1.1, linestyle="--",
                    alpha=0.85, label="Paris +1.5 °C")
        ax8.axhline(2.0, color="#e74c3c", linewidth=1.1, linestyle="--",
                    alpha=0.85, label="Paris +2.0 °C")

    ax8.axvline(df_full["year"].max(), color="#8b949e",
                linewidth=1, linestyle=":", alpha=0.6)
    ax8.text(df_full["year"].max() + 0.5,
             ax8.get_ylim()[0] + 0.05, "▶ Forecast",
             fontsize=8.5, color="#8b949e", alpha=0.8)
    ax8.set_xlabel("Year", fontsize=10)
    ax8.set_ylabel("Temperature Anomaly (°C)", fontsize=10)
    ax8.set_title(
        f"Global Temperature Forecast to {forecast_horizon} — {best_name}\n"
        "LOW = aggressive mitigation  |  MED = business-as-usual  |  HIGH = accelerated emissions",
        fontsize=11, fontweight="bold"
    )
    ax8.legend(fontsize=9, loc="upper left")
    ax8.set_xlim(1880, forecast_horizon + 5)
    fig8.tight_layout()
    st.pyplot(fig8, use_container_width=True)
    plt.close(fig8)

    # Scenario table
    st.markdown('<div class="section-header">Scenario Projections Table</div>',
                unsafe_allow_html=True)
    decade_years = [y for y in future_years if y % 5 == 0]
    proj_rows = []
    for sc_name, preds in scenario_preds.items():
        row = {"Scenario": sc_name}
        for yr in decade_years:
            idx = list(future_years).index(yr) if yr in future_years else None
            if idx is not None:
                row[str(yr)] = f"{preds[idx]:+.3f} °C"
        proj_rows.append(row)
    st.dataframe(pd.DataFrame(proj_rows), use_container_width=True, hide_index=True)

    # Download forecast CSV
    forecast_df = pd.DataFrame({"year": future_years})
    for sc_name, preds in scenario_preds.items():
        forecast_df[f"temp_{sc_name}"] = preds
    csv_bytes = forecast_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇  Download Forecast CSV",
        data=csv_bytes,
        file_name=f"forecast_{forecast_horizon}_data.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">Dataset Explorer</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        search_year = st.number_input("Jump to year", min_value=int(df_full["year"].min()),
                                      max_value=int(df_full["year"].max()),
                                      value=int(df_full["year"].min()), step=1)
    with col_s2:
        sort_by = st.selectbox("Sort by", [TARGET] + FEATURES,
                                format_func=lambda x: FEATURE_LABELS.get(x, x))

    display_df = df_full.sort_values(sort_by, ascending=False).rename(
        columns=FEATURE_LABELS
    )
    st.dataframe(display_df, use_container_width=True, height=400)

    row_yr = df_full[df_full["year"] == search_year]
    if not row_yr.empty:
        st.markdown(f"**Year {search_year} snapshot:**")
        snapshot = row_yr[FEATURES + [TARGET]].T.rename(columns={row_yr.index[0]: "Value"})
        snapshot.index = [FEATURE_LABELS.get(i, i) for i in snapshot.index]
        st.dataframe(snapshot.style.format("{:.4f}"), use_container_width=True)

    st.download_button(
        label="⬇  Download Full Dataset CSV",
        data=df_full.to_csv(index=False).encode(),
        file_name="global_temp_synthetic.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-tag">
  🌡️  Global Temperature Rise ML Dashboard &nbsp;·&nbsp;
  Agbozu Ebingiye Nelvin &nbsp;·&nbsp;
  <a href="https://github.com/Nelvinebi" style="color:#484f58">github.com/Nelvinebi</a>
</div>
""", unsafe_allow_html=True)
