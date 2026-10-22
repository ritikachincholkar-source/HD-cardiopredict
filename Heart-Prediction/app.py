"""CardioPredict - Heart disease risk screening (v2: monotonic Random Forest, no scaler)."""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from fpdf import FPDF

warnings.filterwarnings("ignore")
st.set_page_config(page_title="CardioPredict", page_icon="❤️", layout="centered")

BASE = Path(__file__).parent          # model files must be in the same folder as app.py

# ---------------------------------------------------------------- loading
@st.cache_resource
def load_all():
    return {
        "A": (joblib.load(BASE / "heart_disease_model.pkl"),
              joblib.load(BASE / "heart_disease_columns.pkl")),
        "B": (joblib.load(BASE / "heart_disease_model_no_thal.pkl"),
              joblib.load(BASE / "heart_disease_columns_no_thal.pkl")),
    }

@st.cache_resource
def get_explainer(key, _model):
    return shap.TreeExplainer(_model)

MODELS = load_all()

# ---------------------------------------------------------------- helpers
NAMES = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting blood pressure",
    "chol": "Cholesterol", "fbs": "Fasting blood sugar > 120", "restecg": "Resting ECG",
    "thalach": "Maximum heart rate", "exang": "Exercise-induced angina",
    "oldpeak": "ST depression", "slope": "ST segment slope",
    "ca": "Blocked vessels (fluoroscopy)", "thal": "Thallium stress test",
}

def risk_level(p):
    if p < 0.33:
        return "Low", "#2e9e5b"
    if p < 0.66:
        return "Medium", "#e0a11b"
    return "High", "#c0392b"

def positive_class_shap(explainer, X):
    """Return the SHAP vector (one value per attribute) for the 'disease present' class."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                 # older SHAP: [class0, class1]
        return np.array(sv[1])[0]
    sv = np.array(sv)
    if sv.ndim == 3:                         # newer SHAP: (samples, features, classes)
        return sv[0, :, 1]
    return sv[0]

def suggestions(v, level):
    out = []
    if level in ("Medium", "High"):
        out.append("Consult a cardiologist for further evaluation.")
    if v["chol"] > 240:
        out.append("Cholesterol is elevated: consider dietary changes and a lipid profile test.")
    if v["trestbps"] > 140:
        out.append("Blood pressure is high: regular monitoring is recommended.")
    if v["fbs"] == 1:
        out.append("Screening for diabetes is advised.")
    if v["exang"] == 1:
        out.append("Avoid strenuous activity until evaluated.")
    if not out:
        out.append("No major red flags; maintain a healthy lifestyle and routine check-ups.")
    return out

def gauge(p, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(p * 100, 1), number={"suffix": " %"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 33], "color": "#d5f0df"},
                         {"range": [33, 66], "color": "#fbeccb"},
                         {"range": [66, 100], "color": "#f6d3cf"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10))
    return fig

def clean(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")

def make_pdf(res):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "CardioPredict - Heart Disease Risk Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    def line(txt, bold=False):
        pdf.set_font("Helvetica", "B" if bold else "", 11)
        pdf.multi_cell(0, 7, clean(txt), new_x="LMARGIN", new_y="NEXT")

    line(f"Risk level: {res['level']}", True)
    line(f"Predicted probability: {res['prob'] * 100:.1f} %")
    line(f"Model used: {res['model_name']}")
    line("")
    line("Entered values:", True)
    for k, val in res["display"].items():
        line(f"  {k}: {val}")
    line("")
    line("Main contributing factors (SHAP):", True)
    for name, val, sv in res["top"]:
        direction = "increases" if sv > 0 else "decreases"
        line(f"  {name} = {val}  ->  {direction} risk ({sv:+.3f})")
    line("")
    line("Suggested next steps:", True)
    for s in res["suggest"]:
        line(f"  - {s}")
    line("")
    line("Disclaimer: screening and decision-support aid only. Not a diagnosis. "
         "It does not replace medical advice.")
    return bytes(pdf.output())

# ---------------------------------------------------------------- UI
st.title("❤️ CardioPredict")
st.caption("Heart disease risk screening from routine clinical values. "
           "Educational / screening-support tool, not a diagnosis.")

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age (years)", 1, 120, 50)
    sex_l = st.selectbox("Sex", ["Male", "Female"])
    cp_l = st.selectbox("Chest pain type", ["Typical angina", "Atypical angina",
                                            "Non-anginal pain", "Asymptomatic"])
    trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 120)
    chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 200)
    fbs_l = st.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
    restecg_l = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality",
                                             "Left ventricular hypertrophy"])
with c2:
    thalach = st.number_input("Maximum heart rate achieved (bpm)", 60, 220, 150)
    exang_l = st.selectbox("Exercise-induced angina", ["No", "Yes"])
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 0.0, step=0.1)
    slope_l = st.selectbox("Slope of peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Major vessels coloured by fluoroscopy (0-3)", [0, 1, 2, 3])
    has_thal = st.checkbox("Thallium stress test result is available", value=False)
    thal_l = None
    if has_thal:
        thal_l = st.selectbox("Thallium stress test result",
                              ["Normal", "Fixed defect", "Reversible defect"])

values = {
    "age": age,
    "sex": 1 if sex_l == "Male" else 0,
    "cp": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp_l) + 1,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": 1 if fbs_l == "Yes" else 0,
    "restecg": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg_l),
    "thalach": thalach,
    "exang": 1 if exang_l == "Yes" else 0,
    "oldpeak": oldpeak,
    "slope": ["Upsloping", "Flat", "Downsloping"].index(slope_l) + 1,
    "ca": ca,
}
if has_thal:
    values["thal"] = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal_l]

display = {
    "Age": age, "Sex": sex_l, "Chest pain type": cp_l, "Resting BP (mm Hg)": trestbps,
    "Cholesterol (mg/dl)": chol, "Fasting sugar > 120": fbs_l, "Resting ECG": restecg_l,
    "Max heart rate": thalach, "Exercise angina": exang_l, "ST depression": oldpeak,
    "ST slope": slope_l, "Blocked vessels": ca,
    "Thallium test": thal_l if has_thal else "Not available",
}

def predict(key, vals):
    model, cols = MODELS[key]
    X = pd.DataFrame([[vals[c] for c in cols]], columns=cols)   # same order as training
    return model, X, float(model.predict_proba(X)[0][1])

if st.button("Predict risk", type="primary"):
    key = "A" if has_thal else "B"
    model, X, prob = predict(key, values)
    level, color = risk_level(prob)
    sv = positive_class_shap(get_explainer(key, model), X)
    order = np.argsort(-np.abs(sv))[:5]
    top = [(NAMES[X.columns[i]], X.iloc[0, i], float(sv[i])) for i in order]

    other = None
    if has_thal:                                    # also show the estimate without the thallium test
        _, _, p_b = predict("B", values)
        other = p_b

    st.session_state["res"] = {
        "prob": prob, "level": level, "color": color, "top": top, "display": display,
        "model_name": "Model A (13 attributes, with thallium test)" if has_thal
                      else "Model B (12 attributes, without thallium test)",
        "suggest": suggestions(values, level), "has_thal": has_thal, "p_without_thal": other,
    }

res = st.session_state.get("res")
if res:
    st.divider()
    tag = {"Low": "green", "Medium": "orange", "High": "red"}[res["level"]]
    st.subheader(f"Risk level: :{tag}[{res['level']}]")
    st.caption(f"Model used: {res['model_name']}")
    st.plotly_chart(gauge(res["prob"], res["color"]))

    if res["has_thal"] and res["p_without_thal"] is not None:
        st.caption(f"Estimate without the thallium test for comparison: "
                   f"{res['p_without_thal'] * 100:.1f} %")
    if not res["has_thal"]:
        st.info("Thallium stress test was not provided, so the estimate is less certain. "
                "If the risk is Medium or High, a thallium stress test may help. "
                "A Low result does not rule out disease if you have symptoms; see a doctor.")

    st.subheader("Main contributing factors")
    for name, val, s in res["top"]:
        arrow = "⬆️ raises" if s > 0 else "⬇️ lowers"
        st.write(f"- **{name}** = {val}: {arrow} risk ({s:+.3f})")

    st.subheader("Suggested next steps")
    for s in res["suggest"]:
        st.write(f"- {s}")
    st.caption("These suggestions are general and do not replace medical advice.")

    st.download_button("Download PDF report", data=make_pdf(res),
                       file_name="heart_risk_report.pdf", mime="application/pdf")
