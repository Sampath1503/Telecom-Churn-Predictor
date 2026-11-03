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
st.sidebar.caption("1,000 held-out records")

st.markdown("""
<div class="hero">
 <div class="eyebrow">Customer analytics / Predictive decision support</div>
 <h1>📡 Telecom Churn Intelligence</h1>
 <p>Monitor model performance, assess an individual customer's churn probability, and translate model output into a practical retention follow-up.</p>
</div>
""", unsafe_allow_html=True)

overview, predict, history_tab, methodology = st.tabs(
    ["Overview dashboard", "Customer assessment", "Assessment history", "Model methodology"]
)

with overview:
    st.markdown('<div class="heading">Model performance overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Corrected experiment · Held-out test set · These are model metrics, not live business KPIs.</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    items = [
        ("Test accuracy", "97.7%", "Overall correct classifications"),
        ("Churn precision", "96.8%", "Predicted churners that were churners"),
        ("Churn recall", "86.5%", "Actual churners identified"),
        ("ROC-AUC", "92.5%", "Class separation performance")
    ]
    for col, (label, value, foot) in zip(cols, items):
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1.05, .95], gap="large")
    with c1:
        st.markdown('<div class="panel"><div class="panel-title">Test-set classification summary</div>', unsafe_allow_html=True)
        cm = pd.DataFrame([[855, 4], [19, 122]], index=["Actual: No churn", "Actual: Churn"], columns=["Predicted: No churn", "Predicted: Churn"])
        st.dataframe(cm, use_container_width=True)
        st.caption("The model identified 122 of 141 churn customers and missed 19.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="panel-title">Business interpretation</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Precision:** Fewer customers are incorrectly flagged for retention action.
        - **Recall:** About 86.5% of actual churners were identified in the test set.
        - **Trade-off:** Some churners are still missed; treat scores as decision support, not certainty.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="heading">Assessment activity</div>', unsafe_allow_html=True)
    rows = st.session_state.get("history", [])
    if rows:
        hist = pd.DataFrame(rows)
        a,b,c,d = st.columns(4)
        a.metric("Assessments this session", len(hist))
        b.metric("High risk", int((hist["Risk band"] == "High").sum()))
        c.metric("Moderate risk", int((hist["Risk band"] == "Moderate").sum()))
        d.metric("Low risk", int((hist["Risk band"] == "Low").sum()))
        st.caption("Counts are only for assessments made during this active app session.")
        st.dataframe(hist[["Timestamp", "State", "Churn probability", "Prediction", "Risk band"]], use_container_width=True, hide_index=True)
    else:
        st.info("No customer assessments in this session yet. Open Customer assessment to evaluate a customer.")

with predict:
    st.markdown('<div class="heading">Individual customer assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Enter customer account, plan, usage and service information. The saved pipeline applies its fitted preprocessing automatically.</div>', unsafe_allow_html=True)
    states = ["KS","OH","NJ","OK","AL","MA","MO","LA","WV","IN","RI","IA","MT","NY","ID","VT","VA","TX","FL","CO","AZ","SC","NE","WY","HI","IL","NH","GA","AK","MD","AR","WI","OR","MI","DE","UT","CA","MN","SD","NC","WA","NM","NV","DC","KY","ME","MS","TN","PA","CT","ND"]

    with st.form("customer_form"):
        st.markdown("#### 1. Account profile")
        a,b,c = st.columns(3)
        state = a.selectbox("State", states)
        area_code = b.selectbox("Area code", [408,415,510], index=1)
        account_length = c.number_input("Account length (days)", 1, 250, 100)

        st.markdown("#### 2. Plan information")
        a,b,c = st.columns(3)
        voice_plan = a.selectbox("Voice mail plan", ["no","yes"])
        voice_messages = a.number_input("Voice messages", 0, 60, 0)
        intl_plan = b.selectbox("International plan", ["no","yes"])
        intl_mins = b.number_input("International minutes", 0.0, 60.0, 10.0, .5)
        intl_calls = c.number_input("International calls", 0, 30, 3)
        intl_charge = c.number_input("International charge", 0.0, 20.0, 2.7, .1)

        st.markdown("#### 3. Usage and billing")
        a,b,c = st.columns(3)
        with a:
            st.markdown("**Daytime**")
            day_mins = st.number_input("Day minutes", 0.0, 500.0, 180.0, 1.0)
            day_calls = st.number_input("Day calls", 0, 200, 100)
            day_charge = st.number_input("Day charge", 0.0, 100.0, 30.0, .1)
        with b:
            st.markdown("**Evening**")
            eve_mins = st.number_input("Evening minutes", 0.0, 500.0, 200.0, 1.0)
            eve_calls = st.number_input("Evening calls", 0, 200, 100)
            eve_charge = st.number_input("Evening charge", 0.0, 100.0, 17.0, .1)
        with c:
            st.markdown("**Night**")
            night_mins = st.number_input("Night minutes", 0.0, 500.0, 200.0, 1.0)
            night_calls = st.number_input("Night calls", 0, 200, 100)
            night_charge = st.number_input("Night charge", 0.0, 100.0, 9.0, .1)

        st.markdown("#### 4. Service interaction and decision threshold")
        a,b = st.columns(2)
        customer_calls = a.number_input("Customer service calls", 0, 10, 1)
        threshold = b.slider("Churn classification threshold", .10, .90, .50, .05)
        submitted = st.form_submit_button("Run churn assessment", use_container_width=True)

    if submitted:
        values = {
            "state": state, "area.code": area_code, "account.length": account_length,
            "voice.plan": voice_plan, "voice.messages": voice_messages,
            "intl.plan": intl_plan, "intl.mins": intl_mins, "intl.calls": intl_calls,
            "intl.charge": intl_charge, "day.mins": day_mins, "day.calls": day_calls,
            "day.charge": day_charge, "eve.mins": eve_mins, "eve.calls": eve_calls,
            "eve.charge": eve_charge, "night.mins": night_mins, "night.calls": night_calls,
            "night.charge": night_charge, "customer.calls": customer_calls
        }
        try:
            model_input = make_features(values)
            prob = float(model.predict_proba(model_input)[0, 1])
            record_assessment(values, prob, threshold)
            st.success("Customer assessment completed.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    latest = st.session_state.get("latest")
    if latest:
        prob = latest["prob"]
        band, css, action = risk_info(prob)
        churn = prob >= latest["threshold"]
        st.markdown("---")
        st.markdown("### Assessment result")
        a,b = st.columns([1.1,.9], gap="large")
        with a:
            title = "Likely to churn" if churn else "Not classified as likely to churn"
            st.markdown(f'<div class="status {css}"><div class="eyebrow">{band} risk</div><div class="status-title">{title}</div><div class="status-detail">Classification threshold: {latest["threshold"]:.0%}</div></div>', unsafe_allow_html=True)
            st.write("")
            st.markdown("**Suggested follow-up**")
            st.write(action + " Validate the customer's situation using service history before taking action.")
        with b:
            st.metric("Predicted churn probability", f"{prob:.2%}")
            st.progress(float(np.clip(prob,0,1)))
            st.caption("A model estimate, not a certainty.")
        with st.expander("Review submitted customer details"):
            st.dataframe(pd.DataFrame(list(latest["values"].items()), columns=["Input feature","Value"]), use_container_width=True, hide_index=True)

with history_tab:
    st.markdown('<div class="heading">Assessment history</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Temporary log of predictions made during this active app session.</div>', unsafe_allow_html=True)
    rows = st.session_state.get("history", [])
    if rows:
        hist = pd.DataFrame(rows)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("Download session history (CSV)", hist.to_csv(index=False).encode("utf-8"), "churn_assessment_history.csv", "text/csv")
        if st.button("Clear session history"):
            st.session_state["history"] = []
            st.session_state.pop("latest", None)
            st.rerun()
    else:
        st.info("No assessments recorded yet. Run a prediction in Customer assessment.")

with methodology:
    st.markdown('<div class="heading">Model methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Summary of the corrected training and evaluation process.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
    <div class="panel-title">Training and validation workflow</div>
    <ol>
      <li>Customer feature engineering from account, usage, billing and service-call fields.</li>
      <li>Stratified 80:20 train-test split.</li>
      <li>Fold-specific imputation, encoding and scaling inside the pipeline.</li>
      <li>SMOTETomek applied inside each cross-validation training fold.</li>
      <li>RandomizedSearchCV with 50 candidates and 5-fold stratified cross-validation.</li>
      <li>Final evaluation on the held-out test set.</li>
    </ol></div>
    """, unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Metric":["Accuracy","Churn precision","Churn recall","Churn F1-score","ROC-AUC"],
        "Result":["97.7%","96.8%","86.5%","91.4%","92.5%"]
    }), use_container_width=True, hide_index=True)
    st.warning("Retention guidance is a general suggestion based on probability bands. This app does not yet provide feature attribution or explain why an individual customer may churn.")

st.markdown("---")
st.markdown('<div class="small">Telecom Churn Intelligence · Streamlit · Model metrics refer to the corrected held-out evaluation experiment.</div>', unsafe_allow_html=True)
