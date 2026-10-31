import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# TELECOM CUSTOMER CHURN PREDICTOR
# Streamlit UI for the corrected, leakage-controlled pipeline
# ============================================================

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "churn_xgb_pipeline.pkl"

# ----------------------------
# Styling
# ----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0b1020;
        color: #edf2f7;
    }

    [data-testid="stSidebar"] {
        background: #141b2d;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.7rem 1.9rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 20px;
        background: linear-gradient(135deg, #17243d 0%, #111827 58%, #15243a 100%);
        margin-bottom: 1.3rem;
    }

    .eyebrow {
        color: #60a5fa;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.12;
        font-weight: 800;
        margin: 0 0 0.65rem 0;
    }

    .hero p {
        color: #bdc9dc;
        font-size: 1rem;
        margin: 0;
        max-width: 850px;
    }

    .section-heading {
        color: #e5edf8;
        font-size: 1.12rem;
        font-weight: 700;
        margin: 1.2rem 0 0.7rem 0;
    }

    .metric-card {
        background: #141e31;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 112px;
    }

    .metric-label {
        color: #9fb0c8;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-caption {
        color: #8293ad;
        font-size: 0.76rem;
        margin-top: 0.35rem;
    }

    .result-card {
        border-radius: 18px;
        padding: 1.3rem 1.45rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: #141e31;
        margin-top: 0.7rem;
    }

    .result-low {
        border-left: 5px solid #22c55e;
    }

    .result-medium {
        border-left: 5px solid #f59e0b;
    }

    .result-high {
        border-left: 5px solid #ef4444;
    }

    .result-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .result-subtitle {
        color: #aebdd2;
        font-size: 0.92rem;
    }

    .recommendation {
        background: #172b43;
        border: 1px solid rgba(96, 165, 250, 0.22);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        color: #d7e8fb;
        margin-top: 1rem;
    }

    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        border: 0;
        border-radius: 12px;
        min-height: 3.1rem;
        font-weight: 700;
        font-size: 1rem;
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        color: white;
        width: 100%;
    }

    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        border: 0;
        background: linear-gradient(90deg, #1d4ed8, #0284c7);
        color: white;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        background: #111a2b;
    }

    .small-note {
        color: #8293ad;
        font-size: 0.78rem;
    }

    [data-testid="stMetric"] {
        background: #141e31;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #aebdd2;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.16);
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Load corrected model pipeline
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Model file '{MODEL_PATH}' was not found. "
        "Place the corrected pipeline file in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model pipeline: {exc}")
    st.stop()


# ----------------------------
# Feature engineering
# ----------------------------
def build_customer_dataframe(values: dict) -> pd.DataFrame:
    """Create the same engineered features used during model training."""
    df = pd.DataFrame([values])

    df["total_charge"] = (
        df["day.charge"]
        + df["eve.charge"]
        + df["night.charge"]
        + df["intl.charge"]
    )

    df["total_mins"] = (
        df["day.mins"]
        + df["eve.mins"]
        + df["night.mins"]
        + df["intl.mins"]
    )

    df["charge_per_min"] = df["total_charge"] / (df["total_mins"] + 1e-6)

    df["total_calls"] = (
        df["day.calls"]
        + df["eve.calls"]
        + df["night.calls"]
        + df["intl.calls"]
    )

    df["mins_per_call"] = df["total_mins"] / (df["total_calls"] + 1e-6)

    calls = df["customer.calls"].iloc[0]
    if calls == 0:
        service_bin = "0_calls"
    elif calls <= 3:
        service_bin = "1-3_calls"
    else:
        service_bin = "4+_calls"

    df["service_call_bin"] = service_bin

    df["pay_as_you_go_intl"] = (
        (df["intl.plan"] == "no") & (df["intl.mins"] > 0)
    ).astype(int)

    return df


# ----------------------------
# Sidebar: customer input form
# ----------------------------
states_list = [
    "KS", "OH", "NJ", "OK", "AL", "MA", "MO", "LA", "WV", "IN", "RI",
    "IA", "MT", "NY", "ID", "VT", "VA", "TX", "FL", "CO", "AZ", "SC",
    "NE", "WY", "HI", "IL", "NH", "GA", "AK", "MD", "AR", "WI", "OR",
    "MI", "DE", "UT", "CA", "MN", "SD", "NC", "WA", "NM", "NV", "DC",
    "KY", "ME", "MS", "TN", "PA", "CT", "ND"
]

st.sidebar.markdown("## 👤 Customer profile")
st.sidebar.caption("Enter customer account and usage information.")

with st.sidebar.form("customer_input_form"):
    st.sidebar.markdown("### Account information")
    state = st.selectbox("State", states_list, index=0)
    area_code = st.selectbox("Area code", [408, 415, 510], index=1)
    account_length = st.slider("Account length (days)", 1, 250, 100)

    st.sidebar.markdown("### Voice mail plan")
    voice_plan = st.radio("Voice mail plan", ["yes", "no"], index=1, horizontal=True)
    voice_messages = st.slider("Number of voice messages", 0, 60, 0)

    st.sidebar.markdown("### International plan")
    intl_plan = st.radio("International plan", ["yes", "no"], index=1, horizontal=True)
    intl_mins = st.number_input("International minutes", 0.0, 60.0, 10.0, 0.5)
    intl_calls = st.number_input("International calls", 0, 30, 3)
    intl_charge = st.number_input("International charge", 0.0, 20.0, 2.7, 0.1)

    st.sidebar.markdown("### Daytime usage")
    day_mins = st.number_input("Day minutes", 0.0, 500.0, 180.0, 1.0)
    day_calls = st.number_input("Day calls", 0, 200, 100)
    day_charge = st.number_input("Day charge", 0.0, 100.0, 30.0, 0.1)

    st.sidebar.markdown("### Evening usage")
    eve_mins = st.number_input("Evening minutes", 0.0, 500.0, 200.0, 1.0)
    eve_calls = st.number_input("Evening calls", 0, 200, 100)
    eve_charge = st.number_input("Evening charge", 0.0, 100.0, 17.0, 0.1)

    st.sidebar.markdown("### Night usage")
    night_mins = st.number_input("Night minutes", 0.0, 500.0, 200.0, 1.0)
    night_calls = st.number_input("Night calls", 0, 200, 100)
    night_charge = st.number_input("Night charge", 0.0, 100.0, 9.0, 0.1)

    st.sidebar.markdown("### Customer service")
    customer_calls = st.slider("Customer service calls", 0, 10, 1)

    threshold = st.slider(
        "Churn probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="A customer is classified as likely to churn when the predicted probability is at least this threshold."
    )

    submitted = st.form_submit_button("Predict customer churn", use_container_width=True)

customer_values = {
    "state": state,
    "area.code": area_code,
    "account.length": account_length,
    "voice.plan": voice_plan,
    "voice.messages": voice_messages,
    "intl.plan": intl_plan,
    "intl.mins": intl_mins,
    "intl.calls": intl_calls,
    "intl.charge": intl_charge,
    "day.mins": day_mins,
    "day.calls": day_calls,
    "day.charge": day_charge,
    "eve.mins": eve_mins,
    "eve.calls": eve_calls,
    "eve.charge": eve_charge,
    "night.mins": night_mins,
    "night.calls": night_calls,
    "night.charge": night_charge,
    "customer.calls": customer_calls,
}


# ----------------------------
# Main page header
# ----------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">Customer analytics · Predictive intelligence</div>
    <h1>📡 Telecom Churn Intelligence</h1>
    <p>
        Identify customers at risk of leaving, understand their predicted churn
        probability, and support timely retention decisions using an XGBoost model.
    </p>
</div>
""", unsafe_allow_html=True)

# Model performance is from the corrected held-out test experiment.
st.markdown('<div class="section-heading">Model performance</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Test accuracy", "97.7%")
with m2:
    st.metric("Churn precision", "96.8%")
with m3:
    st.metric("Churn recall", "86.5%")
with m4:
    st.metric("ROC-AUC", "92.5%")

st.caption(
    "Metrics from the corrected experiment on a held-out test set of 1,000 records. "
    "These are model evaluation metrics, not a guarantee for individual predictions."
)

st.divider()

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="section-heading">Customer churn assessment</div>', unsafe_allow_html=True)
    st.write(
        "Enter the customer's information in the sidebar and select "
        "**Predict customer churn** to generate an assessment."
    )

    if submitted:
        try:
            input_df = build_customer_dataframe(customer_values)

            # The saved imblearn pipeline handles preprocessing and prediction.
            probability = float(model.predict_proba(input_df)[0, 1])
            churn_prediction = int(probability >= threshold)

            if probability < 0.20:
                risk_level = "Low"
                risk_class = "result-low"
                recommendation = (
                    "Continue routine customer monitoring. No immediate retention "
                    "intervention is indicated by this model score."
                )
            elif probability < threshold:
                risk_level = "Moderate"
                risk_class = "result-medium"
                recommendation = (
                    "Monitor usage and service interactions. Consider a routine "
                    "customer satisfaction check-in."
                )
            else:
                risk_level = "High"
                risk_class = "result-high"
                recommendation = (
                    "Prioritise a retention review. Contact the customer to understand "
                    "service concerns and assess suitable retention options."
                )

            st.session_state["latest_assessment"] = {
                "probability": probability,
                "prediction": churn_prediction,
                "risk_level": risk_level,
                "risk_class": risk_class,
                "recommendation": recommendation,
                "customer_values": customer_values.copy(),
                "threshold": threshold,
            }

        except Exception as exc:
            st.error(f"Prediction failed. Check the model and input feature compatibility. Details: {exc}")

    assessment = st.session_state.get("latest_assessment")

    if assessment:
        probability_pct = assessment["probability"] * 100
        prediction_text = (
            "Customer is likely to churn"
            if assessment["prediction"] == 1
            else "Customer is unlikely to churn"
        )

        st.markdown(
            f"""
            <div class="result-card {assessment['risk_class']}">
                <div class="eyebrow">Prediction result · {assessment['risk_level']} risk</div>
                <div class="result-title">{prediction_text}</div>
                <div class="result-subtitle">
                    Classification threshold: {assessment['threshold']:.0%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-heading">Churn probability</div>', unsafe_allow_html=True)
        st.progress(min(max(assessment["probability"], 0.0), 1.0))
        st.markdown(f"## {probability_pct:.2f}%")

        st.markdown(
            f"""
            <div class="recommendation">
                <strong>Recommended action</strong><br><br>
                {assessment['recommendation']}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info(
            "Your prediction will appear here. Complete the customer details in "
            "the sidebar and submit the form."
        )

with right:
    st.markdown('<div class="section-heading">How to read the result</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">LOW RISK</div>
        <div class="metric-value" style="color:#4ade80;">Below 20%</div>
        <div class="metric-caption">Routine monitoring is generally appropriate.</div>
    </div>
    <br>
    <div class="metric-card">
        <div class="metric-label">MODERATE RISK</div>
        <div class="metric-value" style="color:#fbbf24;">Below threshold</div>
        <div class="metric-caption">Consider monitoring and a customer check-in.</div>
    </div>
    <br>
    <div class="metric-card">
        <div class="metric-label">HIGH RISK</div>
        <div class="metric-value" style="color:#f87171;">At or above threshold</div>
        <div class="metric-caption">Review the account for possible retention action.</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Risk bands are application display rules. The classification threshold "
        "can be adjusted in the sidebar."
    )

st.divider()

with st.expander("View customer input summary"):
    if "latest_assessment" in st.session_state:
        summary = pd.DataFrame(
            list(st.session_state["latest_assessment"]["customer_values"].items()),
            columns=["Feature", "Value"]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.caption("Run a prediction to view the submitted customer details.")

st.markdown(
    '<p class="small-note">Telecom Customer Churn Prediction · XGBoost · Streamlit</p>',
    unsafe_allow_html=True
)
