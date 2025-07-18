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

# MDCalc style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Clean MDCalc background */
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Inter', sans-serif;
        color: #333333;
    }
    
    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #333333;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #666666;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Section boxes like MDCalc */
    .section-box {
        background: #FFFFFF;
        border: 1px solid #E1E5E9;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333333;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E1E5E9;
    }
    
    /* Radio button styling */
    .stRadio > label {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #333333 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stRadio > div {
        flex-direction: row !important;
        gap: 2rem !important;
    }
    
    .stRadio > div > label {
        background-color: #F8F9FA !important;
        border: 1px solid #DEE2E6 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
    }
    
    .stRadio > div > label:hover {
        background-color: #E9ECEF !important;
        border-color: #ADB5BD !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background-color: #007BFF !important;
        border-color: #007BFF !important;
        color: white !important;
    }
    
    /* Number input styling */
    .stNumberInput > label {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #333333 !important;
    }
    
    .stNumberInput > div > div > input {
        border: 1px solid #DEE2E6 !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
        font-size: 1rem !important;
    }
    
    /* Calculate button like MDCalc */
    .calculate-button {
        background: #28A745;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        width: 100%;
        margin: 2rem 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .calculate-button:hover {
        background: #218838;
    }
    
    /* Results box like MDCalc */
    .results-box {
        background: #28A745;
        color: white;
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    
    .results-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .results-content {
        font-size: 1.2rem;
        line-height: 1.6;
    }
    
    /* Risk levels */
    .risk-high .results-box { background: #DC3545; }
    .risk-medium .results-box { background: #FD7E14; }
    .risk-low .results-box { background: #28A745; }
    
    /* Model info box */
    .info-box {
        background: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .info-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333333;
        margin-bottom: 1rem;
    }
    
    .info-content {
        font-size: 1rem;
        color: #666666;
        line-height: 1.6;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">CPE Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine learning model for predicting CPE colonization risk in ICU patients</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('cpe_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    model_data = load_model()
    
    if model_data is None:
        st.error("Failed to load model")
        return
        
    model = model_data['model']
    features = model_data['features']
    
    # Main layout
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        patient_data = {}
        
        # Part A: Hospital & Demographics
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part A: Hospital & Demographics</div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            patient_data['Hospital days before ICU admission'] = st.number_input(
                "Hospital days before ICU admission", 
                min_value=0, max_value=100, value=5
            )
        with col_b:
            patient_data['Admission to long-term care facility'] = st.radio(
                "Admission from long-term care facility",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="ltc"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Part B: Medical Conditions
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part B: Medical Conditions</div>', unsafe_allow_html=True)
        
        col_c, col_d = st.columns(2)
        with col_c:
            patient_data['ESRD on renal replacement therapy'] = st.radio(
                "ESRD on renal replacement therapy",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="esrd"
            )
            patient_data['VRE'] = st.radio(
                "VRE colonization",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="vre"
            )
        with col_d:
            patient_data['Steroid use'] = st.radio(
                "Steroid use",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="steroid"
            )
            patient_data['Endoscopy'] = st.radio(
                "Recent endoscopy",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="endoscopy"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Part C: Invasive Devices
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part C: Invasive Devices</div>', unsafe_allow_html=True)
        
        col_e, col_f = st.columns(2)
        with col_e:
            patient_data['Central venous catheter'] = st.radio(
                "Central venous catheter",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="cvc"
            )
            patient_data['Nasogastric tube'] = st.radio(
                "Nasogastric tube",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="ngt"
            )
        with col_f:
            patient_data['Biliary drain'] = st.radio(
                "Biliary drain",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="biliary"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Part D: Antibiotic Exposure
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part D: Antibiotic Exposure</div>', unsafe_allow_html=True)
        
        col_g, col_h = st.columns(2)
        with col_g:
            patient_data['Carbapenem'] = st.radio(
                "Carbapenem use",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="carbapenem"
            )
            patient_data['Aminoglycoside'] = st.radio(
                "Aminoglycoside use",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="amino"
            )
        with col_h:
            patient_data['Antibiotic_Risk'] = st.number_input(
                "Antibiotic Risk Score",
                min_value=0, max_value=10, value=2
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate button
        if st.button("Calculate CPE Risk", key="calc_btn"):
            input_df = pd.DataFrame([patient_data])
            input_df = input_df[features]
            
            try:
                probability = model.predict_proba(input_df)[0, 1]
                st.session_state.probability = probability
                st.session_state.show_result = True
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    with col2:
        # Results section
        if hasattr(st.session_state, 'show_result') and st.session_state.show_result:
            probability = st.session_state.probability
            
            # Determine risk level
            if probability >= 0.45:
                risk_class = "risk-high"
                risk_text = "HIGH RISK"
                recommendation = "Consider CPE isolation precautions and targeted screening"
            elif probability >= 0.3:
                risk_class = "risk-medium"
                risk_text = "MODERATE RISK"
                recommendation = "Enhanced monitoring and standard infection control measures"
            else:
                risk_class = "risk-low"
                risk_text = "LOW RISK"
                recommendation = "Standard care protocols appropriate"
            
            # Results box like MDCalc
            st.markdown(f'''
            <div class="{risk_class}">
                <div class="results-box">
                    <div class="results-title">CPE Risk Assessment</div>
                    <div class="results-content">
                        <strong>{risk_text}</strong><br>
                        Risk Probability: <strong>{probability*100:.1f}%</strong><br><br>
                        {recommendation}
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Additional metrics
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('<div class="info-title">Model Performance</div>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="info-content">
                • <strong>Model Threshold:</strong> 45.0%<br>
                • <strong>ROC-AUC:</strong> 0.774<br>
                • <strong>Sensitivity:</strong> 72.5%<br>
                • <strong>Specificity:</strong> 68.9%<br>
                • <strong>Training Data:</strong> 3,932 patients
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Model information
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('<div class="info-title">About This Calculator</div>', unsafe_allow_html=True)
            st.markdown('''
            <div class="info-content">
                This machine learning model predicts the risk of CPE (Carbapenem-resistant Enterobacteriaceae) colonization in ICU patients based on clinical variables.<br><br>
                
                <strong>Algorithm:</strong> Logistic Regression<br>
                <strong>Features:</strong> 12 clinical variables<br>
                <strong>Validation:</strong> Temporal validation (2022→2023)<br>
                <strong>Performance:</strong> ROC-AUC 0.774
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
