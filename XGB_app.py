import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="Telecom Churn Intelligence", page_icon="📡", layout="wide")

MODEL_PATH = "churn_xgb_pipeline.pkl"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif}
.stApp {background:#0b1220;color:#e8eef7}
[data-testid="stSidebar"] {background:#101a2a;border-right:1px solid #26354a}
.main .block-container {max-width:1450px;padding-top:1.5rem}
.hero {background:linear-gradient(120deg,#172b48,#142033 60%,#10243a);border:1px solid #29394f;border-radius:20px;padding:1.6rem 1.8rem;margin-bottom:1.2rem}
.eyebrow {color:#7db7ff;font-size:.72rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase}
.hero h1 {color:#f8fafc;font-size:2.55rem;font-weight:800;margin:.45rem 0}
.hero p {color:#b6c5d8;margin:0;line-height:1.6}
.heading {color:#edf4ff;font-size:1.2rem;font-weight:800;margin:.5rem 0}
.sub {color:#94a3b8;font-size:.86rem;margin-bottom:1rem}
.kpi {background:linear-gradient(180deg,#172438,#121d2e);border:1px solid #29394f;border-radius:15px;padding:1rem;min-height:112px}
.kpi-label {color:#9eb0c7;font-size:.78rem;font-weight:700}
.kpi-value {color:#f8fafc;font-size:1.85rem;font-weight:800;margin-top:.35rem}
.kpi-foot {color:#8295ad;font-size:.72rem;margin-top:.3rem}
.panel {background:#121d2e;border:1px solid #29394f;border-radius:16px;padding:1.1rem 1.2rem}
.panel-title {color:#e8eef7;font-weight:800;margin-bottom:.6rem}
.status {border-radius:15px;padding:1.1rem 1.2rem;border:1px solid #29394f;background:#142236}
.low {border-left:5px solid #22c55e}.medium {border-left:5px solid #f59e0b}.high {border-left:5px solid #ef4444}
.status-title {color:#f8fafc;font-size:1.25rem;font-weight:800}
.status-detail {color:#aebed2;font-size:.85rem;margin-top:.3rem}
div.stButton>button,div[data-testid="stFormSubmitButton"]>button {border:0;border-radius:11px;min-height:2.9rem;font-weight:800;background:linear-gradient(90deg,#2563eb,#0284c7);color:white;width:100%}
[data-testid="stMetric"] {background:#172438;border:1px solid #29394f;border-radius:13px;padding:.8rem}
div[data-testid="stTabs"] button {font-weight:700}
hr {border-color:#29394f}
.small {color:#8295ad;font-size:.75rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Missing {MODEL_PATH}. Upload it beside this app in GitHub.")
    st.stop()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

def make_features(values):
    df = pd.DataFrame([values])
    df["total_charge"] = df["day.charge"] + df["eve.charge"] + df["night.charge"] + df["intl.charge"]
    df["total_mins"] = df["day.mins"] + df["eve.mins"] + df["night.mins"] + df["intl.mins"]
    df["charge_per_min"] = df["total_charge"] / (df["total_mins"] + 1e-6)
    df["total_calls"] = df["day.calls"] + df["eve.calls"] + df["night.calls"] + df["intl.calls"]
    df["mins_per_call"] = df["total_mins"] / (df["total_calls"] + 1e-6)
    calls = df["customer.calls"].iloc[0]
    df["service_call_bin"] = "0_calls" if calls == 0 else ("1-3_calls" if calls <= 3 else "4+_calls")
    df["pay_as_you_go_intl"] = ((df["intl.plan"] == "no") & (df["intl.mins"] > 0)).astype(int)
    return df

def risk_info(prob):
    if prob < .20:
        return "Low", "low", "Routine monitoring. No immediate retention intervention is indicated by this score."
    if prob < .50:
        return "Moderate", "medium", "Review at the next customer touchpoint and consider a satisfaction check-in."
    return "High", "high", "Prioritise a retention review. Contact the customer to understand service concerns."

def record_assessment(values, prob, threshold):
    band, _, action = risk_info(prob)
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "State": values["state"],
        "Account length": values["account.length"],
        "Service calls": values["customer.calls"],
        "Churn probability": round(prob, 5),
        "Prediction": "Churn" if prob >= threshold else "No churn",
        "Risk band": band,
        "Threshold": threshold,
        "Suggested follow-up": action
    }
    st.session_state.setdefault("history", []).insert(0, row)
    st.session_state["latest"] = {"values": values.copy(), "prob": prob, "threshold": threshold}

# Sidebar branding
st.sidebar.markdown("## 📡 Churn Intelligence")
st.sidebar.caption("TELECOM CUSTOMER ANALYTICS")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model**")
st.sidebar.caption("Corrected XGBoost pipeline")
st.sidebar.markdown("**Evaluation sample**")
