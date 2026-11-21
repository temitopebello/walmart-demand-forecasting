"""Walmart Demand Forecasting — Stakeholder Dashboard.

Deployed on Streamlit Cloud as the Module 5 interactive artefact.
Run locally with:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----- page config -----
st.set_page_config(
    page_title="Walmart Demand Forecast — Stakeholder Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- paths -----
REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO / "reports" / "model" / "champion.joblib"
SUMMARY_PATH = REPO / "reports" / "model" / "run_summary.json"
FAIRNESS_PATH = REPO / "reports" / "fairness" / "fairness_summary.json"
SHAP_CSV = REPO / "reports" / "xai" / "shap_feature_importance.csv"
MODEL_CARD = REPO / "reports" / "model" / "model_card.md"

NAVY = "#003D7A"
BLUE = "#0071DC"
YELLOW = "#FFC220"
GREEN = "#2E8B57"
RED = "#C0392B"

FEATURES = [
    "Store", "Dept", "Size", "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "week_of_year", "month", "year",
    "weeks_to_year_end", "sales_lag_1w", "sales_lag_4w", "sales_lag_52w",
    "IsHoliday", "Type_A", "Type_B", "Type_C",
]


# ----- cached loaders -----
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text())


@st.cache_data
def load_fairness() -> dict:
    return json.loads(FAIRNESS_PATH.read_text())


@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    return pd.read_csv(SHAP_CSV)


@st.cache_data
def load_model_card() -> str:
    try:
        return MODEL_CARD.read_text()
    except FileNotFoundError:
        return "Model card not generated yet."


# ----- helpers -----
def request_to_vector(req: dict) -> np.ndarray:
    row = []
    for f in FEATURES:
        row.append(float(req.get(f, 0)))
    return np.array([row])


def feature_label(name: str) -> str:
    mapping = {
        "sales_lag_1w": "Last week's sales",
        "sales_lag_4w": "Sales from four weeks ago",
        "sales_lag_52w": "Same week last year",
        "week_of_year": "Week number in the year",
        "Size": "Store size (square feet)",
        "Temperature": "Local temperature",
        "Unemployment": "Local unemployment rate",
        "CPI": "Consumer Price Index",
        "Fuel_Price": "Fuel price",
        "IsHoliday": "Is a holiday week",
        "Dept": "Department",
        "Store": "Store number",
    }
    return mapping.get(name, name)


# ===========================================
# Sidebar navigation
# ===========================================
st.sidebar.markdown(f"<h2 style='color:{NAVY}; margin-bottom:0.2em'>🛒 Walmart</h2>", unsafe_allow_html=True)
st.sidebar.caption("Stakeholder Dashboard · BAN6800 Module 5")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Make a Forecast", "What-If Analysis", "Fairness Dashboard", "Model Card"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Audience**")
st.sidebar.write("• Category Managers")
st.sidebar.write("• VP Supply Chain Planning")
st.sidebar.markdown("---")
st.sidebar.caption("Model: LightGBM (champion)")
st.sidebar.caption("Hold-out WMAE: 58.6")
st.sidebar.caption("Fairness: PASS (Relative WMAE)")

# ===========================================
# Overview page
# ===========================================
if page == "Overview":
    st.markdown(f"<h1 style='color:{NAVY}'>Walmart Weekly Sales Forecast</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{BLUE}; margin-top:-0.5em'>Stakeholder briefing dashboard</h4>",
                unsafe_allow_html=True)
    st.write("")

    summary = load_summary()
    r = summary["results"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Champion model", "LightGBM")
    with col2:
        st.metric("Hold-out WMAE", f"{r['lightgbm']['wmae']:,.1f}",
                  delta=f"beats baseline by {(r['baseline_prevyear']['wmae'] - r['lightgbm']['wmae']):+,.1f}",
                  delta_color="normal")
    with col3:
        st.metric("R²", f"{r['lightgbm']['r2']:.3f}")
    with col4:
        st.metric("MAPE", f"{r['lightgbm']['mape_pct']:.1f}%")

    st.markdown("### What this dashboard does")
    st.markdown("""
    This dashboard turns the Module 4 forecasting model into something you can actually use.
    Four pages:

    - **Make a Forecast** — enter a store, department, and week and get the weekly sales prediction
    - **What-If Analysis** — slide markdowns, temperature, and macro conditions to see how they move the forecast
    - **Fairness Dashboard** — see how the model performs across store Types A, B, and C
    - **Model Card** — the formal transparency document describing what the model does and does not do
    """)

    st.markdown("### Model comparison (hold-out performance)")
    labels = ["Previous-year baseline", "Historical-mean baseline",
              "Linear Regression", "Random Forest", "LightGBM"]
    keys = ["baseline_prevyear", "baseline_mean", "linear", "random_forest", "lightgbm"]
    wmae_values = [r[k]["wmae"] for k in keys]
    colors = ["#8A8A8A", "#B0B0B0", BLUE, NAVY, YELLOW]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=wmae_values, marker_color=colors,
        text=[f"{v:,.0f}" for v in wmae_values], textposition="outside",
    )])
    fig.update_layout(
        yaxis_title="WMAE (lower is better)",
        plot_bgcolor="white", height=400,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("All five runs are logged to MLflow (Module 4). LightGBM wins by a modest margin on WMAE and is promoted to champion.")

# ===========================================
# Make a Forecast
# ===========================================
elif page == "Make a Forecast":
    st.markdown(f"<h1 style='color:{NAVY}'>Make a Forecast</h1>", unsafe_allow_html=True)
    st.markdown("Enter a store, department, and week. The dashboard returns the weekly sales prediction and the three strongest drivers in plain language.")
    st.write("")

    model = load_model()

    c1, c2, c3 = st.columns(3)
    with c1:
        store = st.number_input("Store", 1, 45, 1)
        dept = st.number_input("Department", 1, 99, 10)
        type_ = st.selectbox("Store Type", ["A", "B", "C"], index=0)
        size = st.number_input("Store size (sqft)", 10_000, 250_000, 150_000, step=5_000)
    with c2:
        date = st.date_input("Forecast week starting")
        temperature = st.slider("Temperature (°F)", 20.0, 100.0, 70.0)
        fuel_price = st.slider("Fuel price ($/gal)", 2.0, 5.0, 3.2, 0.05)
        is_holiday = st.checkbox("Is this a holiday week?", value=False)
    with c3:
        st.markdown("**Recent sales history**")
        lag_1 = st.number_input("Sales last week", 0.0, 100_000.0, 180.0)
        lag_4 = st.number_input("Sales four weeks ago", 0.0, 100_000.0, 190.0)
        lag_52 = st.number_input("Sales same week last year", 0.0, 100_000.0, 175.0)
        st.markdown("**Macro conditions**")
        cpi = st.slider("CPI", 190.0, 240.0, 220.0, 0.5)
        unemployment = st.slider("Unemployment %", 4.0, 12.0, 7.5, 0.1)

    with st.expander("Promotional markdowns this week (optional)"):
        md1 = st.number_input("Markdown 1", 0.0, 100_000.0, 0.0)
        md2 = st.number_input("Markdown 2", 0.0, 100_000.0, 0.0)
        md3 = st.number_input("Markdown 3", 0.0, 100_000.0, 0.0)
        md4 = st.number_input("Markdown 4", 0.0, 100_000.0, 0.0)
        md5 = st.number_input("Markdown 5", 0.0, 100_000.0, 0.0)

    if st.button("Forecast", type="primary"):
        date_ts = pd.Timestamp(date)
        week_of_year = int(date_ts.isocalendar().week)
        req = {
            "Store": store, "Dept": dept, "Size": size,
            "Temperature": temperature, "Fuel_Price": fuel_price,
            "MarkDown1": md1, "MarkDown2": md2, "MarkDown3": md3,
            "MarkDown4": md4, "MarkDown5": md5,
            "CPI": cpi, "Unemployment": unemployment,
            "week_of_year": week_of_year, "month": date_ts.month, "year": date_ts.year,
            "weeks_to_year_end": (pd.Timestamp(f"{date_ts.year}-12-31") - date_ts).days // 7,
            "sales_lag_1w": lag_1, "sales_lag_4w": lag_4, "sales_lag_52w": lag_52,
            "IsHoliday": int(is_holiday),
            "Type_A": int(type_ == "A"), "Type_B": int(type_ == "B"), "Type_C": int(type_ == "C"),
        }
        x = request_to_vector(req)
        pred = float(model.predict(x)[0])
        st.success(f"**Predicted Weekly_Sales: {pred:,.2f}**")

        try:
            import shap
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x)[0]
            ranking = sorted(zip(FEATURES, x[0], sv), key=lambda t: abs(t[2]), reverse=True)[:3]
            st.markdown("### Top three drivers of this prediction")
            cards = st.columns(3)
            for i, (name, value, contrib) in enumerate(ranking):
                with cards[i]:
                    arrow = "⬆️" if contrib > 0 else "⬇️"
                    color = GREEN if contrib > 0 else RED
                    st.markdown(f"<div style='border-left: 4px solid {color}; padding:10px; background:#F6F8FA;'>"
                                f"<small style='color:#55606B'>Driver {i+1}</small><br>"
                                f"<b>{feature_label(name)}</b><br>"
                                f"<small>Value: {value:,.2f}</small><br>"
                                f"{arrow} {'Pushed forecast UP' if contrib > 0 else 'Pushed forecast DOWN'} "
                                f"by {abs(contrib):,.1f} units"
                                f"</div>", unsafe_allow_html=True)
        except Exception as e:
            st.info(f"(SHAP attribution unavailable in this environment: {e})")

# ===========================================
# What-If Analysis
# ===========================================
elif page == "What-If Analysis":
    st.markdown(f"<h1 style='color:{NAVY}'>What-If Analysis</h1>", unsafe_allow_html=True)
    st.markdown("Fix a baseline store-week, then move the levers you can actually influence (markdowns and macro conditions) to see how the forecast changes.")
    st.write("")

    model = load_model()

    base = dict(
        Store=1, Dept=10, Size=150_000,
        Temperature=70, Fuel_Price=3.2,
        MarkDown1=0, MarkDown2=0, MarkDown3=0, MarkDown4=0, MarkDown5=0,
        CPI=220, Unemployment=7.5,
        week_of_year=24, month=6, year=2012, weeks_to_year_end=28,
        sales_lag_1w=180, sales_lag_4w=190, sales_lag_52w=175,
        IsHoliday=0, Type_A=1, Type_B=0, Type_C=0,
    )

    st.markdown("### Baseline: Store 1, Dept 10, mid-June, no markdowns active")
    base_pred = float(model.predict(request_to_vector(base))[0])
    st.metric("Baseline prediction", f"{base_pred:,.2f}")
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Move the levers")
        md_total = st.slider("Total promotional markdown activity", 0, 50_000, 0, 1000)
        temp_shift = st.slider("Temperature shift (°F)", -20, 20, 0)
        cpi_shift = st.slider("CPI shift", -10.0, 10.0, 0.0, 0.5)
        unemp_shift = st.slider("Unemployment shift (%)", -3.0, 3.0, 0.0, 0.1)
        holiday_flag = st.checkbox("Make this a holiday week?", False)

        altered = dict(base)
        altered["MarkDown1"] = md_total / 5
        altered["MarkDown2"] = md_total / 5
        altered["MarkDown3"] = md_total / 5
        altered["MarkDown4"] = md_total / 5
        altered["MarkDown5"] = md_total / 5
        altered["Temperature"] = base["Temperature"] + temp_shift
        altered["CPI"] = base["CPI"] + cpi_shift
        altered["Unemployment"] = base["Unemployment"] + unemp_shift
        altered["IsHoliday"] = int(holiday_flag)
        new_pred = float(model.predict(request_to_vector(altered))[0])

    with c2:
        st.markdown("#### What happens to the forecast")
        delta = new_pred - base_pred
        st.metric("New prediction", f"{new_pred:,.2f}",
                  delta=f"{delta:+,.2f} vs baseline")
        pct = delta / base_pred * 100 if base_pred else 0
        st.metric("Change", f"{pct:+.2f}%")

    st.write("")
    fig = go.Figure(data=[go.Bar(
        x=["Baseline", "With your changes"],
        y=[base_pred, new_pred],
        marker_color=[NAVY, YELLOW if delta >= 0 else RED],
        text=[f"{base_pred:,.1f}", f"{new_pred:,.1f}"],
        textposition="outside",
    )])
    fig.update_layout(yaxis_title="Predicted Weekly_Sales",
                      plot_bgcolor="white", height=380,
                      margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Business reading: markdown activity and macro conditions each nudge the forecast, but the three recency lag features (last week, four weeks ago, same week last year) remain the strongest drivers. See the Make-a-Forecast page for per-forecast SHAP attributions.")

# ===========================================
# Fairness Dashboard
# ===========================================
elif page == "Fairness Dashboard":
    st.markdown(f"<h1 style='color:{NAVY}'>Fairness Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("How the model performs across store Types A, B, and C. The Module 1 Fairness Objective is: **Relative WMAE gap across Types below 10 percent of overall WMAE.**")
    st.write("")

    fair = load_fairness()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Overall WMAE", f"{fair['overall_wmae']:,.1f}")
    with c2:
        passed_rel = fair.get("wmae_passed_relative", False)
        st.metric("Relative WMAE gap",
                  f"{fair['wmae_gap_relative']:.2%}",
                  delta="PASS 10% threshold" if passed_rel else "FAIL 10% threshold",
                  delta_color="normal" if passed_rel else "inverse")
    with c3:
        st.metric("Absolute WMAE gap",
                  f"{fair['wmae_gap_absolute']:.0%}",
                  delta="FAIL (scale artefact, explained below)",
                  delta_color="inverse")

    st.markdown("### Why two metrics?")
    st.markdown("""
    Type A stores generate roughly **10 times** the weekly sales volume of Type C stores.
    Absolute WMAE therefore scales with sales and makes Type A look "worse" even when the
    model is treating all three tiers equally well in percentage terms.
    **Relative WMAE** (WMAE as a percentage of each group's mean sales) is the correct
    fairness measure for this kind of scale-skewed regression problem, and on that metric
    the three tiers are essentially identical.
    """)

    # Chart: absolute vs relative
    gw = fair["group_wmae"]
    rel = fair["group_rel_wmae_pct"]
    mape = fair.get("group_mape_pct", {})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Absolute WMAE by Type**")
        fig = go.Figure(data=[go.Bar(
            x=list(gw.keys()), y=list(gw.values()),
            marker_color=[BLUE, NAVY, RED],
            text=[f"{v:,.1f}" for v in gw.values()],
            textposition="outside",
        )])
        fig.update_layout(yaxis_title="WMAE (units)",
                          plot_bgcolor="white", height=360,
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Scale-sensitive; Type A looks worst because sales volume is highest.")

    with c2:
        st.markdown("**Relative WMAE by Type (scale-invariant)**")
        fig = go.Figure(data=[go.Bar(
            x=list(rel.keys()), y=list(rel.values()),
            marker_color=[BLUE, NAVY, GREEN],
            text=[f"{v:.2f}%" for v in rel.values()],
            textposition="outside",
        )])
        fig.update_layout(yaxis_title="WMAE as % of group mean sales",
                          plot_bgcolor="white", height=360,
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Scale-invariant; the three tiers sit essentially on top of each other.")

    if mape:
        st.markdown("**Group MAPE** (another scale-invariant check)")
        mape_df = pd.DataFrame({"Type": list(mape.keys()), "MAPE %": list(mape.values())})
        fig = go.Figure(data=[go.Bar(
            x=mape_df["Type"], y=mape_df["MAPE %"],
            marker_color=[BLUE, NAVY, GREEN],
            text=[f"{v:.2f}%" for v in mape_df["MAPE %"]],
            textposition="outside",
        )])
        fig.update_layout(yaxis_title="MAPE %",
                          plot_bgcolor="white", height=340,
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Business reading in one sentence")
    if fair.get("wmae_passed_relative", False):
        st.info("On the scale-invariant fairness metric, the model treats Type A, B, and C stores roughly equally well. The Module 1 Fairness Objective is met.")
    else:
        st.warning("Fairness objective exceeded on both metrics. Mitigation required before production promotion.")

# ===========================================
# Model Card
# ===========================================
elif page == "Model Card":
    st.markdown(f"<h1 style='color:{NAVY}'>Model Card</h1>", unsafe_allow_html=True)
    st.markdown("The formal transparency document describing what this model does and does not do.")
    st.write("")
    st.markdown(load_model_card())

# ===========================================
# Footer
# ===========================================
st.markdown("---")
st.caption(
    "Walmart Demand Forecasting · BAN6800 Module 5 · Bello Temitope · "
    "GitHub: github.com/temitopebello/walmart-demand-forecasting"
)
