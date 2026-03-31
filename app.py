"""
================================================================================
  Streamlit Dashboard — Predicting Global Temperature Rise Using ML Models
  Author : Agbozu Ebingiye Nelvin
  GitHub : https://github.com/Nelvinebi
  Email  : nelvinebingiye@gmail.com
================================================================================
  Run:  streamlit run app.py
  Deps: streamlit plotly pandas numpy scikit-learn openpyxl joblib
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib
import os

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Temperature Rise — ML Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #080e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1628 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; font-size: 0.9rem; }

/* Hide default header */
#MainMenu, footer, header { visibility: hidden; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0f1f3d 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #e74c3c, #f39c12, #27ae60);
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
}
.metric-delta {
    font-size: 0.85rem;
    color: #22c55e;
    margin-top: 4px;
    font-weight: 500;
}

/* Page hero */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}
.accent { color: #e74c3c; }
.accent-blue { color: #38bdf8; }
.accent-green { color: #22c55e; }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-left: 3px solid #e74c3c;
    padding-left: 12px;
    margin: 28px 0 16px 0;
}

/* Info pill */
.pill {
    display: inline-block;
    background: #1e3a5f;
    color: #7dd3fc;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px;
    border: 1px solid #2d5480;
}

/* Scenario badge */
.badge-low    { background:#0f2d1a; color:#4ade80; border:1px solid #16643a; border-radius:6px; padding:4px 10px; font-size:0.82rem; font-weight:600; }
.badge-med    { background:#2d1f0a; color:#fbbf24; border:1px solid #5c3d0a; border-radius:6px; padding:4px 10px; font-size:0.82rem; font-weight:600; }
.badge-high   { background:#2d0f0f; color:#f87171; border:1px solid #5c1f1f; border-radius:6px; padding:4px 10px; font-size:0.82rem; font-weight:600; }

/* Divider */
.divider { border: none; border-top: 1px solid #1e2d4a; margin: 24px 0; }

/* Plotly chart background override */
.js-plotly-plot .plotly { background: transparent !important; }

/* Streamlit overrides */
.stSelectbox > div > div { background: #0d1628 !important; border: 1px solid #1e3a5f !important; }
.stSlider > div > div > div { background: #1e3a5f !important; }
div[data-testid="stMetric"] { background: #0f1f3d; border-radius: 10px; padding: 12px 16px; border: 1px solid #1e3a5f; }
div[data-testid="stMetric"] label { color: #64748b !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-family: 'Space Mono', monospace; font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL  (cached)
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    "co2_ppm", "ch4_ppb", "n2o_ppb", "aerosol_optical_depth",
    "solar_irradiance_anom", "enso_index", "volcanic_forcing",
    "land_use_index", "urbanization_index",
]
FEATURE_LABELS = {
    "co2_ppm":               "CO₂ (ppm)",
    "ch4_ppb":               "CH₄ (ppb)",
    "n2o_ppb":               "N₂O (ppb)",
    "aerosol_optical_depth": "Aerosol Depth",
    "solar_irradiance_anom": "Solar Irradiance",
    "enso_index":            "ENSO Index",
    "volcanic_forcing":      "Volcanic Forcing",
    "land_use_index":        "Land Use Index",
    "urbanization_index":    "Urbanisation Index",
}
TARGET   = "temp_anomaly_C"
PALETTE  = {"Ridge": "#38bdf8", "Random Forest": "#22c55e", "Gradient Boosting": "#f97316"}
SC_COL   = {"LOW": "#22c55e", "MED": "#fbbf24", "HIGH": "#ef4444"}

@st.cache_data
def load_data():
    paths = [
        "Data/global_temp_synthetic.xlsx",
        "global_temp_synthetic.xlsx",
        "data/global_temp_synthetic.xlsx",
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_excel(p)
    st.error("Dataset not found. Place `global_temp_synthetic.xlsx` in the `Data/` folder.")
    st.stop()

@st.cache_resource
def train_models(df):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

    mdls = {
        "Ridge":             Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                        max_depth=4, random_state=42, subsample=0.8),
    }
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, m in mdls.items():
        m.fit(Xtr_s, y_tr)
        yp = m.predict(Xte_s)
        cv = cross_val_score(m, scaler.transform(X), y, cv=kf, scoring="r2", n_jobs=-1)
        results[name] = {
            "model": m, "y_pred": yp, "y_test": y_te,
            "MAE":  mean_absolute_error(y_te, yp),
            "RMSE": np.sqrt(mean_squared_error(y_te, yp)),
            "R2":   r2_score(y_te, yp),
            "CV_mean": cv.mean(), "CV_std": cv.std(),
        }
    best = max(results, key=lambda n: results[n]["R2"])
    return results, scaler, best, X_tr, X_te, y_tr, y_te, Xtr_s, Xte_s

def build_scenario(base, years, co2_rate, ch4_rate, n2o_rate, aerosol_trend, urban_rate, seed=42):
    np.random.seed(seed)
    rows = []
    for i, _ in enumerate(years):
        t = i + 1
        rows.append({
            "co2_ppm":              base["co2_ppm"]              + co2_rate * t,
            "ch4_ppb":              base["ch4_ppb"]              + ch4_rate * t,
            "n2o_ppb":              base["n2o_ppb"]              + n2o_rate * t,
            "aerosol_optical_depth":base["aerosol_optical_depth"] + aerosol_trend * t,
            "solar_irradiance_anom":base["solar_irradiance_anom"] + np.random.normal(0, 0.02),
            "enso_index":           np.random.normal(0, 0.5),
            "volcanic_forcing":     np.random.normal(-0.05, 0.05),
            "land_use_index":       min(base["land_use_index"]   + 0.005 * t, 1.0),
            "urbanization_index":   min(base["urbanization_index"] + urban_rate * t, 1.0),
        })
    return pd.DataFrame(rows)

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data()
results, scaler, best_name, X_tr, X_te, y_tr, y_te, Xtr_s, Xte_s = train_models(df)
best_model = results[best_name]["model"]
base_2024  = df[df["year"] == df["year"].max()][FEATURES].iloc[0].to_dict()
future_years = np.arange(2025, 2051)

scenarios_data = {
    "LOW":  build_scenario(base_2024, future_years, 0.5,  0.3,  0.1,  0.002, 0.004),
    "MED":  build_scenario(base_2024, future_years, 2.5,  1.5,  0.3,  0.0,   0.008),
    "HIGH": build_scenario(base_2024, future_years, 4.5,  3.0,  0.6, -0.002, 0.012),
}
scenario_preds = {
    sc: best_model.predict(scaler.transform(df_[FEATURES]))
    for sc, df_ in scenarios_data.items()
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0'>
      <div style='font-family:Space Mono,monospace;font-size:1.05rem;color:#f1f5f9;font-weight:700;line-height:1.3'>
        🌡️ Global Temp<br>Rise — ML
      </div>
      <div style='font-size:0.75rem;color:#475569;margin-top:4px'>Environmental Data Science</div>
    </div>
    <hr style='border-color:#1e2d4a;margin:12px 0'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview", "📊  EDA & Trends", "🤖  Model Performance",
         "🔮  2050 Forecast", "🎛️  Scenario Simulator", "📋  Data Explorer"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e2d4a;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;color:#334155;line-height:1.8'>
      <div style='color:#475569;font-weight:600;margin-bottom:6px;font-size:0.78rem'>AUTHOR</div>
      <div>Agbozu Ebingiye Nelvin</div>
      <div style='margin-top:4px'>
        <a href='https://github.com/Nelvinebi' style='color:#38bdf8;text-decoration:none'>GitHub</a> ·
        <a href='https://www.linkedin.com/in/agbozu-ebi/' style='color:#38bdf8;text-decoration:none'>LinkedIn</a>
      </div>
      <div style='margin-top:2px'>
        <a href='mailto:nelvinebingiye@gmail.com' style='color:#38bdf8;text-decoration:none'>nelvinebingiye@gmail.com</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div class='hero-title'>🌡️ Predicting <span class='accent'>Global Temperature Rise</span></div>
    <div class='hero-sub'>Machine Learning · Climate Forcing · 2050 Scenario Forecasting · Port Harcourt, Nigeria</div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("R² Score", f"{results[best_name]['R2']:.4f}", "Best model accuracy"),
        ("MAE", f"{results[best_name]['MAE']:.4f}°C", "Mean absolute error"),
        ("RMSE", f"{results[best_name]['RMSE']:.4f}°C", "Root mean squared error"),
        ("HIGH 2050", f"{scenario_preds['HIGH'][-1]:.2f}°C", "High-emission projection"),
        ("LOW 2050",  f"{scenario_preds['LOW'][-1]:.2f}°C",  "Low-emission projection"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4,c5], kpis):
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{val}</div>
          <div class='metric-label'>{label}</div>
          <div style='font-size:0.72rem;color:#475569;margin-top:4px'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Mini forecast chart + key facts side by side
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("<div class='section-header'>Historical trend & 2050 forecast</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["year"], y=df[TARGET],
            mode="lines", name="Historical",
            line=dict(color="#94a3b8", width=1.5),
            fill="tozeroy", fillcolor="rgba(148,163,184,0.06)"
        ))
        for sc, col_ in SC_COL.items():
            fig.add_trace(go.Scatter(
                x=list(future_years), y=list(scenario_preds[sc]),
                mode="lines", name=f"{sc} scenario",
                line=dict(color=col_, width=2.2),
                fill="tozeroy",
                fillcolor=f"rgba({int(col_[1:3],16)},{int(col_[3:5],16)},{int(col_[5:7],16)},0.07)"
            ))
        fig.add_hline(y=1.5, line_dash="dot", line_color="#fbbf24", annotation_text="Paris +1.5°C", annotation_font_color="#fbbf24")
        fig.add_hline(y=2.0, line_dash="dot", line_color="#ef4444", annotation_text="Paris +2.0°C", annotation_font_color="#ef4444")
        fig.add_vline(x=2024, line_dash="dash", line_color="#475569")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,22,40,0.6)",
            font=dict(family="DM Sans", color="#94a3b8"),
            margin=dict(l=0, r=0, t=10, b=0), height=300,
            legend=dict(orientation="h", y=1.05, font_size=11, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1e2d4a", tickfont_size=11),
            yaxis=dict(gridcolor="#1e2d4a", tickfont_size=11, title="Temp Anomaly (°C)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Project facts</div>", unsafe_allow_html=True)
        facts = [
            ("📅", "Coverage", "1880 – 2024"),
            ("📊", "Dataset", "145 years × 11 cols"),
            ("🔬", "Features", "9 climate-forcing vars"),
            ("🤖", "Models", "Ridge · RF · GB"),
            ("🏆", "Best model", best_name),
            ("🌍", "Region", "Global mean"),
        ]
        for icon, lbl, val in facts:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;
                        background:#0d1628;border:1px solid #1e2d4a;
                        border-radius:8px;padding:10px 14px;margin-bottom:7px'>
              <span style='font-size:1.1rem'>{icon}</span>
              <div>
                <div style='font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em'>{lbl}</div>
                <div style='font-size:0.88rem;color:#cbd5e1;font-weight:500'>{val}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Feature tags
    st.markdown("<div class='section-header'>Climate-forcing features used</div>", unsafe_allow_html=True)
    tags_html = "".join(f"<span class='pill'>{v}</span>" for v in FEATURE_LABELS.values())
    st.markdown(f"<div style='line-height:2.2'>{tags_html}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown("<div class='hero-title'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Climate forcing variables · historical trends · correlations</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌡️ Temperature Trend", "🔗 Correlation Matrix", "📈 Feature Trends"])

    with tab1:
        st.markdown("<div class='section-header'>Global mean temperature anomaly — 1880 to 2024</div>", unsafe_allow_html=True)
        rolling = df.set_index("year")[TARGET].rolling(10, center=True).mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["year"], y=df[TARGET], mode="lines", name="Annual",
            line=dict(color="rgba(239,68,68,0.5)", width=1),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.07)"
        ))
        fig.add_trace(go.Scatter(
            x=rolling["year"], y=rolling[TARGET], mode="lines",
            name="10-yr rolling mean", line=dict(color="#f97316", width=2.5)
        ))
        fig.add_hline(y=1.5, line_dash="dot", line_color="#fbbf24",
                      annotation_text="Paris +1.5°C", annotation_font_color="#fbbf24")
        fig.add_hline(y=2.0, line_dash="dot", line_color="#ef4444",
                      annotation_text="Paris +2.0°C", annotation_font_color="#ef4444")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,22,40,0.6)",
            font=dict(family="DM Sans", color="#94a3b8"),
            margin=dict(l=0, r=0, t=10, b=0), height=380,
            legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a", title="Temperature Anomaly (°C)"),
        )
        st.plotly_chart(fig, use_container_width=True)
        # Stats row
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Min anomaly",  f"{df[TARGET].min():.3f} °C")
        s2.metric("Max anomaly",  f"{df[TARGET].max():.3f} °C")
        s3.metric("Mean anomaly", f"{df[TARGET].mean():.3f} °C")
        s4.metric("2024 anomaly", f"{df[df['year']==df['year'].max()][TARGET].values[0]:.3f} °C")

    with tab2:
        st.markdown("<div class='section-header'>Pearson correlation matrix</div>", unsafe_allow_html=True)
        corr = df[FEATURES + [TARGET]].corr()
        labels = [FEATURE_LABELS.get(f, f) for f in FEATURES] + ["Temp Anomaly"]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=labels, y=labels,
            colorscale=[[0,"#1e3a5f"],[0.5,"#0d1628"],[1,"#991b1b"]],
            zmid=0, text=np.round(corr.values, 2),
            texttemplate="%{text}", textfont_size=10,
            hoverongaps=False,
            colorbar=dict(tickfont=dict(color="#94a3b8"), outlinewidth=0)
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,22,40,0.6)",
            font=dict(family="DM Sans", color="#94a3b8"),
            margin=dict(l=0, r=0, t=10, b=0), height=480,
            xaxis=dict(tickangle=-35, tickfont_size=11),
            yaxis=dict(tickfont_size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-header'>Individual feature trends over time</div>", unsafe_allow_html=True)
        feat_sel = st.selectbox("Select feature", options=list(FEATURE_LABELS.keys()),
                                format_func=lambda x: FEATURE_LABELS[x])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["year"], y=df[feat_sel], mode="lines",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
            name=FEATURE_LABELS[feat_sel]
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,22,40,0.6)",
            font=dict(family="DM Sans", color="#94a3b8"),
            margin=dict(l=0, r=0, t=10, b=0), height=340,
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a", title=FEATURE_LABELS[feat_sel]),
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown("<div class='hero-title'>🤖 Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Ridge · Random Forest · Gradient Boosting — side-by-side evaluation</div>", unsafe_allow_html=True)

    # Metrics table
    st.markdown("<div class='section-header'>Evaluation metrics — all models</div>", unsafe_allow_html=True)
    metrics_df = pd.DataFrame([
        {
            "Model": name,
            "MAE (°C)": f"{r['MAE']:.4f}",
            "RMSE (°C)": f"{r['RMSE']:.4f}",
            "R² Score": f"{r['R2']:.4f}",
            "CV R² (5-fold)": f"{r['CV_mean']:.4f} ± {r['CV_std']:.4f}",
            "Status": "🏆 Best" if name == best_name else "—"
        }
        for name, r in results.items()
    ])
    st.dataframe(
        metrics_df, use_container_width=True, hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(width="small"),
            "R² Score": st.column_config.TextColumn(width="small"),
        }
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Actual vs Predicted — all 3 models
    st.markdown("<div class='section-header'>Actual vs predicted — test set</div>", unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(results.keys()),
                        shared_yaxes=True)
    for i, (name, r) in enumerate(results.items(), 1):
        test_years = df["year"].iloc[X_te.index].values
        color = PALETTE[name]
        fig.add_trace(go.Scatter(
            x=test_years, y=r["y_test"].values,
            mode="lines", name="Actual",
            line=dict(color="#94a3b8", width=1.5),
            showlegend=(i == 1)
        ), row=1, col=i)
        fig.add_trace(go.Scatter(
            x=test_years, y=r["y_pred"],
            mode="lines", name=f"{name} predicted",
            line=dict(color=color, width=2, dash="dash"),
            showlegend=True
        ), row=1, col=i)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,22,40,0.6)",
        font=dict(family="DM Sans", color="#94a3b8"),
        margin=dict(l=0, r=0, t=40, b=0), height=320,
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor="#1e2d4a")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Feature importance
    st.markdown("<div class='section-header'>Permutation feature importance — best model</div>", unsafe_allow_html=True)
    with st.spinner("Computing permutation importance…"):
        perm = permutation_importance(
            best_model, scaler.transform(X_te), y_te,
            n_repeats=20, random_state=42, n_jobs=-1
        )
    feat_imp = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURES],
        "Importance": perm.importances_mean,
        "Std": perm.importances_std,
    }).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        y=feat_imp["Feature"],
        x=feat_imp["Importance"],
        orientation="h",
        error_x=dict(array=feat_imp["Std"].tolist(), color="#475569"),
        marker_color=["#ef4444" if v > 0 else "#334155" for v in feat_imp["Importance"]],
        marker_line_width=0,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,22,40,0.6)",
        font=dict(family="DM Sans", color="#94a3b8"),
        margin=dict(l=0, r=0, t=10, b=0), height=340,
        xaxis=dict(gridcolor="#1e2d4a", title="Mean Permutation Importance (Δ R²)"),
        yaxis=dict(gridcolor="#1e2d4a"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residual analysis
    st.markdown("<div class='section-header'>Residual analysis — best model</div>", unsafe_allow_html=True)
    residuals = results[best_name]["y_test"].values - results[best_name]["y_pred"]
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Scatter(
            x=results[best_name]["y_pred"], y=residuals,
            mode="markers",
            marker=dict(color=PALETTE[best_name], size=8, opacity=0.8,
                        line=dict(color="#0d1628", width=0.5))
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#475569")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,22,40,0.6)",
            font=dict(family="DM Sans", color="#94a3b8"),
            margin=dict(l=0, r=0, t=10, b=0), height=280,
            xaxis=dict(gridcolor="#1e2d4a", title="Predicted (°C)"),
            yaxis=dict(gridcolor="#1e2d4a", title="Residual (°C)"),
            title=dict(text="Residuals vs Predicted", font_color="#94a3b8", font_size=13)
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Histogram(
            x=residuals, nbinsx=15,
            marker_color=PALETTE[best_name],
            marker_line_color="#0d1628", marker_line_width=1,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#475569")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,22,40,0.6)",
            font=dict(family="DM Sans", color="#94a3b8"),
            margin=dict(l=0, r=0, t=10, b=0), height=280,
            xaxis=dict(gridcolor="#1e2d4a", title="Residual (°C)"),
            yaxis=dict(gridcolor="#1e2d4a", title="Count"),
            title=dict(text="Residual Distribution", font_color="#94a3b8", font_size=13)
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — 2050 FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif "2050" in page:
    st.markdown("<div class='hero-title'>🔮 2050 Temperature Forecast</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Three emission pathways · Paris Agreement thresholds · endpoint projections</div>", unsafe_allow_html=True)

    # Scenario summary cards
    c1, c2, c3 = st.columns(3)
    scenario_info = {
        "LOW":  ("🟢 LOW scenario", "Aggressive mitigation", "net-zero pathway", "#22c55e", "badge-low"),
        "MED":  ("🟡 MED scenario", "Business as usual",    "moderate action",  "#fbbf24", "badge-med"),
        "HIGH": ("🔴 HIGH scenario","Accelerated emissions", "no policy change", "#ef4444", "badge-high"),
    }
    for col_, (sc, (title, sub1, sub2, color, badge)) in zip([c1,c2,c3], scenario_info.items()):
        col_.markdown(f"""
        <div style='background:#0d1628;border:1px solid {color}33;border-radius:12px;
                    padding:20px;border-top:3px solid {color};text-align:center'>
          <div style='font-size:1rem;color:#f1f5f9;font-weight:600'>{title}</div>
          <div style='font-size:0.78rem;color:#64748b;margin:4px 0'>{sub1} · {sub2}</div>
          <div style='font-family:Space Mono,monospace;font-size:2rem;
                      color:{color};font-weight:700;margin:10px 0'>
            {scenario_preds[sc][-1]:.2f}°C
          </div>
          <div style='font-size:0.72rem;color:#475569'>projected anomaly by 2050</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Full forecast chart — 1880 to 2050</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year"], y=df[TARGET], mode="lines", name="Historical (1880–2024)",
        line=dict(color="#94a3b8", width=1.5),
        fill="tozeroy", fillcolor="rgba(148,163,184,0.05)"
    ))
    for sc, color in SC_COL.items():
        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fig.add_trace(go.Scatter(
            x=list(future_years), y=list(scenario_preds[sc]),
            mode="lines", name=f"{sc} scenario",
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.08)"
        ))
        fig.add_annotation(
            x=2050, y=scenario_preds[sc][-1],
            text=f" {sc}: {scenario_preds[sc][-1]:.2f}°C",
            showarrow=False, xanchor="left",
            font=dict(color=color, size=11, family="Space Mono")
        )
    fig.add_hline(y=1.5, line_dash="dot", line_color="#fbbf24", line_width=1.2,
                  annotation_text="Paris +1.5°C", annotation_font_color="#fbbf24",
                  annotation_position="bottom right")
    fig.add_hline(y=2.0, line_dash="dot", line_color="#ef4444", line_width=1.2,
                  annotation_text="Paris +2.0°C", annotation_font_color="#ef4444",
                  annotation_position="bottom right")
    fig.add_vline(x=2024, line_dash="dash", line_color="#334155", line_width=1)
    fig.add_annotation(x=2026, y=df[TARGET].min()+0.1,
                       text="← Historical  |  Forecast →",
                       showarrow=False, font=dict(color="#475569", size=11))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,22,40,0.6)",
        font=dict(family="DM Sans", color="#94a3b8"),
        margin=dict(l=0, r=80, t=10, b=0), height=420,
        legend=dict(orientation="h", y=1.06, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1e2d4a", range=[1880, 2055]),
        yaxis=dict(gridcolor="#1e2d4a", title="Temperature Anomaly (°C)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Forecast data table — 2025 to 2050</div>", unsafe_allow_html=True)
    forecast_df = pd.DataFrame({
        "Year": future_years,
        "LOW (°C)":  [round(v,3) for v in scenario_preds["LOW"]],
        "MED (°C)":  [round(v,3) for v in scenario_preds["MED"]],
        "HIGH (°C)": [round(v,3) for v in scenario_preds["HIGH"]],
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True, height=280)
    st.download_button(
        "⬇️  Download forecast CSV",
        forecast_df.to_csv(index=False).encode(),
        file_name="forecast_2050.csv", mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SCENARIO SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Simulator" in page:
    st.markdown("<div class='hero-title'>🎛️ Scenario Simulator</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Adjust emission rates and instantly see the projected 2050 temperature outcome</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Custom emission controls</div>", unsafe_allow_html=True)

    col_ctrl, col_result = st.columns([1, 2])

    with col_ctrl:
        co2_rate   = st.slider("CO₂ annual increase (ppm/yr)",   0.0, 6.0, 2.5, 0.1)
        ch4_rate   = st.slider("CH₄ annual increase (ppb/yr)",   0.0, 5.0, 1.5, 0.1)
        n2o_rate   = st.slider("N₂O annual increase (ppb/yr)",   0.0, 1.0, 0.3, 0.05)
        urban_rate = st.slider("Urbanisation rate (index/yr)",   0.0, 0.02, 0.008, 0.001)
        aerosol    = st.slider("Aerosol trend (depth/yr)",       -0.005, 0.005, 0.0, 0.001)

    with col_result:
        custom_sc = build_scenario(
            base_2024, future_years,
            co2_rate, ch4_rate, n2o_rate, aerosol, urban_rate, seed=99
        )
        custom_preds = best_model.predict(scaler.transform(custom_sc[FEATURES]))
        end_temp = custom_preds[-1]

        # Color based on severity
        if end_temp < 2.5:
            temp_color = "#22c55e"
        elif end_temp < 3.2:
            temp_color = "#fbbf24"
        else:
            temp_color = "#ef4444"

        st.markdown(f"""
        <div style='background:#0d1628;border:1px solid {temp_color}55;border-radius:16px;
                    padding:28px;text-align:center;border-top:4px solid {temp_color};margin-bottom:16px'>
          <div style='font-size:0.78rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em'>
            Projected temperature anomaly by 2050
          </div>
          <div style='font-family:Space Mono,monospace;font-size:3.5rem;
                      color:{temp_color};font-weight:700;line-height:1.1;margin:12px 0'>
            {end_temp:.2f}°C
          </div>
          <div style='font-size:0.85rem;color:#64748b'>
            {"✅ Below Paris +2.0°C" if end_temp < 2.0 else "⚠️ Above Paris +2.0°C" if end_temp < 3.0 else "🚨 Critically above Paris targets"}
          </div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["year"], y=df[TARGET], mode="lines", name="Historical",
            line=dict(color="#94a3b8", width=1.2)
        ))
        for sc, color in SC_COL.items():
            fig.add_trace(go.Scatter(
                x=list(future_years), y=list(scenario_preds[sc]),
                mode="lines", name=sc,
                line=dict(color=color, width=1.2, dash="dot"),
                opacity=0.5
            ))
        fig.add_trace(go.Scatter(
            x=list(future_years), y=list(custom_preds),
            mode="lines", name="Your scenario",
            line=dict(color=temp_color, width=3),
            fill="tozeroy", fillcolor=f"rgba({int(temp_color[1:3],16)},{int(temp_color[3:5],16)},{int(temp_color[5:7],16)},0.1)"
        ))
        fig.add_hline(y=1.5, line_dash="dot", line_color="#fbbf24", line_width=1)
        fig.add_hline(y=2.0, line_dash="dot", line_color="#ef4444", line_width=1)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,22,40,0.6)",
            font=dict(family="DM Sans", color="#94a3b8"),
            margin=dict(l=0, r=0, t=10, b=0), height=300,
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)", font_size=11),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a", title="Temp Anomaly (°C)"),
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Data" in page:
    st.markdown("<div class='hero-title'>📋 Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Browse and filter the full synthetic climate dataset</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        year_range = st.slider("Year range", int(df["year"].min()),
                               int(df["year"].max()),
                               (int(df["year"].min()), int(df["year"].max())))
        show_feat = st.multiselect(
            "Columns to display",
            options=["year"] + FEATURES + [TARGET],
            default=["year", "co2_ppm", "temp_anomaly_C"]
        )

    filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    with col2:
        s1, s2, s3 = st.columns(3)
        s1.metric("Rows shown", len(filtered))
        s2.metric("CO₂ range", f"{filtered['co2_ppm'].min():.1f} – {filtered['co2_ppm'].max():.1f} ppm")
        s3.metric("Temp range", f"{filtered[TARGET].min():.2f} – {filtered[TARGET].max():.2f} °C")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    if show_feat:
        st.dataframe(
            filtered[show_feat].reset_index(drop=True),
            use_container_width=True, height=420
        )
        st.download_button(
            "⬇️  Download filtered data as CSV",
            filtered[show_feat].to_csv(index=False).encode(),
            file_name=f"climate_data_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv",
        )

    st.markdown("<div class='section-header'>Descriptive statistics</div>", unsafe_allow_html=True)
    st.dataframe(
        filtered[FEATURES + [TARGET]].describe().round(4),
        use_container_width=True
    )
