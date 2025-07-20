### ✅ 수정 요약
# - 모든 공백을 유발하는 `st.markdown("<div>")` 와 `</div>` 를 하나로 묶어서 최소화
# - 빈 공간 생기는 원인 제거 (특히 열고 바로 닫는 div 제거)
# - CSS 에 .stMarkdown, .block-container 에 여백 제거 코드 추가

# 전체 수정 코드는 아래 캔버스에 반영됩니다.

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="CPE Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply CSS fix to remove unwanted vertical spacing
st.markdown("""
<style>
/* Remove default top/bottom padding */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}

/* Remove spacing above and below markdown */
.stMarkdown {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

/* Remove empty containers that can create white space */
.element-container:empty {
    display: none !important;
    height: 0px !important;
    margin: 0px !important;
    padding: 0px !important;
}

/* Fix spacing inside radio/number inputs */
div[data-testid="stRadio"] > label,
div[data-testid="stNumberInput"] > label {
    margin-bottom: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 style="text-align:center; font-size: 3rem; margin-bottom: 1rem; font-style:italic;">CPE (Carbapenemase-producing Enterobacterales) Risk Predictor</h1>', unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        with open('cpe_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

# Validation function
def validate_inputs(patient_data, antibiotic_data):
    warnings = []
    if patient_data.get('ESRD on renal replacement therapy') == 1 and patient_data.get('Hospital days before ICU admission') == 0:
        warnings.append("Patient on RRT with 0 hospital days - please verify")
    if sum(antibiotic_data.values()) >= 4:
        warnings.append("High antibiotic exposure (4+ classes) - increased CPE risk")
    if patient_data.get('Central venous catheter') == 1 and patient_data.get('Nasogastric tube') == 1 and patient_data.get('Biliary drain') == 1:
        warnings.append("Multiple invasive devices present - consider enhanced monitoring")
    return warnings

def main():
    model_data = load_model()
    if model_data is None:
        return
    model = model_data['model']
    features = model_data['features']

    col1, col2 = st.columns([2.5, 1], gap="large")

    with col1:
        patient_data = {}
        antibiotic_data = {}

        with st.container():
            st.subheader("Part A: Healthcare Exposure & Admission History")
            c1, c2 = st.columns(2)
            with c1:
                patient_data['Hospital days before ICU admission'] = st.number_input("Hospital days before ICU admission", min_value=0, max_value=365, value=0)
            with c2:
                patient_data['Admission to long-term care facility'] = st.radio("Admission from long-term care facility", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)

        with st.container():
            st.subheader("Part B: Medical Conditions & Interventions")
            c1, c2 = st.columns(2)
            with c1:
                patient_data['ESRD on renal replacement therapy'] = st.radio("ESRD on renal replacement therapy", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)
                patient_data['VRE'] = st.radio("VRE colonization history", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)
            with c2:
                patient_data['Steroid use'] = st.radio("Recent steroid use", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)
                patient_data['Endoscopy'] = st.radio("Recent endoscopy", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)

        with st.container():
            st.subheader("Part C: Invasive Devices")
            c1, c2, c3 = st.columns(3)
            with c1:
                patient_data['Central venous catheter'] = st.radio("Central venous catheter", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)
            with c2:
                patient_data['Nasogastric tube'] = st.radio("Nasogastric tube", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)
            with c3:
                patient_data['Biliary drain'] = st.radio("Biliary drain", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)

        with st.container():
            st.subheader("Part D: Antibiotic Exposure")
            d1, d2 = st.columns(2)
            with d1:
                antibiotic_data['Fluoroquinolone'] = st.radio("Fluoroquinolone", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)
                antibiotic_data['Cephalosporin'] = st.radio("Cephalosporin", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)
                antibiotic_data['Carbapenem'] = st.radio("Carbapenem", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)
            with d2:
                antibiotic_data['β-lactam/β-lactamase inhibitor'] = st.radio("Beta-lactam/BLI", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)
                antibiotic_data['Aminoglycoside'] = st.radio("Aminoglycoside", [0, 1], format_func=lambda x: "YES" if x else "NO", horizontal=True)

            score = sum(antibiotic_data.values())
            st.markdown(f"<div class='auto-calc'>Antibiotic Risk Score: <strong>{score}</strong></div>", unsafe_allow_html=True)

        for warning in validate_inputs(patient_data, antibiotic_data):
            st.warning(warning)

        if st.button("🔬 CALCULATE CPE RISK", use_container_width=True):
            model_input = {
                'Hospital days before ICU admission': patient_data['Hospital days before ICU admission'],
                'ESRD on renal replacement therapy': patient_data['ESRD on renal replacement therapy'],
                'Steroid use': patient_data['Steroid use'],
                'Central venous catheter': patient_data['Central venous catheter'],
                'Nasogastric tube': patient_data['Nasogastric tube'],
                'Biliary drain': patient_data['Biliary drain'],
                'Carbapenem': antibiotic_data['Carbapenem'],
                'Aminoglycoside': antibiotic_data['Aminoglycoside'],
                'Admission to long-term care facility': patient_data['Admission to long-term care facility'],
                'VRE': patient_data['VRE'],
                'Endoscopy': patient_data['Endoscopy'],
                'Antibiotic_Risk': score
            }
            df = pd.DataFrame([model_input])[features]
            try:
                prob = model.predict_proba(df)[0, 1]
                st.session_state.probability = prob
                st.session_state.show_result = True
                st.success("Risk assessment completed successfully!")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    with col2:
        if st.session_state.get("show_result"):
            p = st.session_state.probability
            if p >= 0.45:
                label, color = "HIGH RISK", "#DC2626"
                msg = "Immediate CPE screening and isolation precautions recommended"
            elif p >= 0.3:
                label, color = "MODERATE RISK", "#F59E0B"
                msg = "Enhanced monitoring and standard precautions"
            else:
                label, color = "LOW RISK", "#10B981"
                msg = "Standard infection control measures"

            st.markdown(f"""
            <div class='results-box' style='background:{color};'>
                <div class='results-title'>🎯 CPE Risk Assessment</div>
                <div class='results-content'>
                    <strong>{label}</strong><br>
                    Risk Probability: <strong>{p*100:.1f}%</strong><br><br>
                    {msg}
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
