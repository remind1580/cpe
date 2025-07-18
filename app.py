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

# Professional medical UI with elegant fonts
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400;1,600&family=Lato:ital,wght@0,400;0,700;1,400;1,700&display=swap');
    
    /* Clean medical background */
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Lato', sans-serif;
        color: #1F2937;
    }
    
    /* Elegant main title with italic serif */
    .main-title {
        font-family: 'Crimson Text', serif;
        font-size: 4rem;
        font-weight: 600;
        color: #1a365d;
        text-align: center;
        margin-bottom: 4rem;
        font-style: italic;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    
    /* Section boxes with more spacing */
    .section-box {
        background: #FFFFFF;
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 3rem;
        margin: 2.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Elegant section headers with italic */
    .section-header {
        font-family: 'Lato', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c5282;
        font-style: italic;
        text-transform: capitalize;
        letter-spacing: 0.3px;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    /* Large radio button labels with italic */
    .stRadio > label {
        font-family: 'Lato', sans-serif !important;
        font-size: 1.6rem !important;        /* 1.4rem → 1.6rem */
        font-weight: 600 !important;         /* 500 → 600 더 진하게 */
        color: #000000 !important;           /* #1F2937 → #000000 순검정 */
        margin-bottom: 1rem !important;
        line-height: 1.6 !important;
        font-style: normal !important;       /* italic 제거 */
    }
    
    /* Enhanced radio buttons with more spacing */
    .stRadio > div {
        flex-direction: row !important;
        gap: 3rem !important;
        margin-top: 1rem !important;
        margin-bottom: 2rem !important;
    }
    
    .stRadio > div > label {
        background-color: #F9FAFB !important;
        border: 3px solid #D1D5DB !important;    /* 2px → 3px */
        border-radius: 10px !important;          /* 8px → 10px */
        padding: 1.2rem 2.5rem !important;       /* 1rem 2rem → 1.2rem 2.5rem */
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        font-size: 1.4rem !important;            /* 1.2rem → 1.4rem */
        font-weight: 700 !important;             /* 600 → 700 */
    }
    
    .stRadio > div > label:hover {
        background-color: #EBF4FF !important;
        border-color: #3B82F6 !important;
        transform: translateY(-1px) !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background-color: #3B82F6 !important;
        border-color: #3B82F6 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Large number input labels */
    .stNumberInput > label {
        font-family: 'Lato', sans-serif !important;
        font-size: 1.6rem !important;        /* 1.4rem → 1.6rem */
        font-weight: 600 !important;         /* 500 → 600 */
        color: #000000 !important;           /* 순검정 */
        line-height: 1.6 !important;
        font-style: normal !important;       /* italic 제거 */
        margin-bottom: 1rem !important;
    }
    
    .stNumberInput > div > div > input {
        border: 2px solid #D1D5DB !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-size: 1.2rem !important;
        background-color: #F9FAFB !important;
        font-family: 'Lato', sans-serif !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Time frame indicators - larger */
    .time-frame {
        font-size: 1.2rem;                       /* 1rem → 1.2rem */
        color: #374151;                          /* #6B7280 → #374151 더 진하게 */
        font-style: italic;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-family: 'Lato', sans-serif;
        font-weight: 500;     
    }
    
    /* Drug examples - larger */
    .drug-examples {
        font-size: 1.1rem;                       /* 0.95rem → 1.1rem */
        color: #374151;                          /* #4B5563 → #374151 */
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-style: italic;
        font-family: 'Lato', sans-serif;
        font-weight: 500;      
    }
    
    /* Calculate button */
    .stButton > button {
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1.5rem 4rem !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3) !important;
        text-transform: uppercase !important;
        width: 100% !important;
        font-family: 'Lato', sans-serif !important;
        margin-top: 2rem !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669, #047857) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Results boxes */
    .results-box {
        background: #10B981;
        color: white;
        border-radius: 12px;
        padding: 3rem;
        margin: 2.5rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .results-title {
        font-family: 'Crimson Text', serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .results-content {
        font-family: 'Lato', sans-serif;
        font-size: 1.5rem;
        line-height: 1.6;
    }
    
    .risk-high .results-box { background: linear-gradient(135deg, #DC2626, #B91C1C); }
    .risk-medium .results-box { background: linear-gradient(135deg, #D97706, #B45309); }
    .risk-low .results-box { background: linear-gradient(135deg, #10B981, #059669); }
    
    /* Info boxes */
    .info-box {
        background: #F8FAFC;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 2.5rem;
        margin: 2rem 0;
    }
    
    .info-title {
        font-family: 'Crimson Text', serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .info-content {
        font-family: 'Lato', sans-serif;
        font-size: 1.2rem;
        color: #4B5563;
        line-height: 1.7;
    }
    
    /* Auto-calculated score */
    .auto-calc {
        background: linear-gradient(135deg, #EBF8FF, #DBEAFE);
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 1.5rem;
        font-size: 1.4rem;
        color: #1E40AF;
        text-align: center;
        margin-top: 2rem;
        font-weight: 600;
        font-family: 'Lato', sans-serif;
    }
    
    /* Validation warnings */
    .validation-warning {
        background: #FEF2F2;
        border: 2px solid #F87171;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #B91C1C;
        font-size: 1.1rem;
        font-weight: 500;
        font-family: 'Lato', sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
</style>
""", unsafe_allow_html=True)

# Elegant title with italic serif
st.markdown('<h1 class="main-title">CPE (Carbapenemase-producing Enterobacterales) Risk Predictor</h1>', unsafe_allow_html=True)
@st.cache_resource
def load_model():
    try:
        with open('cpe_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def validate_inputs(patient_data, antibiotic_data):
    warnings = []
    
    # ESRD validation
    if patient_data.get('ESRD on renal replacement therapy') == 0:
        warnings.append("⚠️ Consider if patient truly has no renal replacement therapy")
    
    # Antibiotic exposure validation
    antibiotic_count = sum(antibiotic_data.values())
    if antibiotic_count >= 4:
        warnings.append("⚠️ High antibiotic exposure detected - please verify")
    
    return warnings

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
        antibiotic_data = {}
        
        # Part A: Healthcare Exposure & Admission History
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part A: Healthcare Exposure & Admission History</div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            patient_data['Hospital days before ICU admission'] = st.number_input(
                "Hospital days before ICU admission", 
                min_value=0, max_value=100, value=0,
                help="Number of days hospitalized in this facility before ICU transfer"
            )
            st.markdown('<div class="time-frame">Count days from current admission to ICU transfer</div>', unsafe_allow_html=True)
            
        with col_b:
            patient_data['Admission to long-term care facility'] = st.radio(
                "Admission from long-term care facility",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="ltc"
            )
            st.markdown('<div class="time-frame">Nursing home, Long-Term Acute Care, or rehabilitation facility</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Part B: Medical Conditions & Interventions
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part B: Medical Conditions & Interventions</div>', unsafe_allow_html=True)
        
        col_c, col_d = st.columns(2)
        with col_c:
            patient_data['ESRD on renal replacement therapy'] = st.radio(
                "ESRD on renal replacement therapy",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="esrd"
            )
            st.markdown('<div class="time-frame">Dialysis or CRRT required</div>', unsafe_allow_html=True)
            
            patient_data['VRE'] = st.radio(
                "VRE colonization",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="vre"
            )
            st.markdown('<div class="time-frame">Previous VRE colonization (any time)</div>', unsafe_allow_html=True)
            
        with col_d:
            patient_data['Steroid use'] = st.radio(
                "Steroid use",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="steroid"
            )
            st.markdown('<div class="time-frame">Systemic corticosteroids within 3 months</div>', unsafe_allow_html=True)
            
            patient_data['Endoscopy'] = st.radio(
                "Recent endoscopy",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="endoscopy"
            )
            st.markdown('<div class="time-frame">Any endoscopic procedure within 30 days</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Part C: Invasive Devices
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part C: Invasive Devices</div>', unsafe_allow_html=True)
        st.markdown('<div class="time-frame" style="margin-bottom: 1rem; font-size: 1rem; color: #6B7280;">⏱️ Within 48 hours of ICU admission</div>', unsafe_allow_html=True)
        
        col_e, col_f = st.columns(2)
        with col_e:
            patient_data['Central venous catheter'] = st.radio(
                "Central venous catheter",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="cvc"
            )
            st.markdown('<div class="drug-examples">Subclavian, jugular, femoral line</div>', unsafe_allow_html=True)
            
            patient_data['Nasogastric tube'] = st.radio(
                "Nasogastric tube",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="ngt"
            )
            st.markdown('<div class="drug-examples">NG tube or OG tube placement</div>', unsafe_allow_html=True)
            
        with col_f:
            patient_data['Biliary drain'] = st.radio(
                "Biliary drain",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="biliary"
            )
            st.markdown('<div class="drug-examples">PTBD, ERCP stent, drainage procedures</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Part D: Antibiotic Exposure
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part D: Antibiotic Exposure</div>', unsafe_allow_html=True)
        st.markdown('<div class="time-frame" style="margin-bottom: 1.5rem; font-size: 1rem; color: #6B7280;">⏱️ Within 3 months before ICU admission</div>', unsafe_allow_html=True)
        
        col_g, col_h = st.columns(2)
        with col_g:
            antibiotic_data['Fluoroquinolone'] = st.radio(
                "Fluoroquinolone",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="fluoro"
            )
            st.markdown('<div class="drug-examples">Ciprofloxacin, Levofloxacin, Moxifloxacin</div>', unsafe_allow_html=True)
            
            antibiotic_data['Cephalosporin'] = st.radio(
                "Cephalosporin",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="ceph"
            )
            st.markdown('<div class="drug-examples">Ceftriaxone, Cefazolin, Ceftazidime, Cefepime</div>', unsafe_allow_html=True)
            
            antibiotic_data['Carbapenem'] = st.radio(
                "Carbapenem",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="carbap"
            )
            st.markdown('<div class="drug-examples">Meropenem, Imipenem, Ertapenem, Doripenem</div>', unsafe_allow_html=True)
            
        with col_h:
            antibiotic_data['β-lactam/β-lactamase inhibitor'] = st.radio(
                "β-lactam/β-lactamase inhibitor",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="beta_lactam"
            )
            st.markdown('<div class="drug-examples">Piperacillin/Tazobactam, Ampicillin/Sulbactam</div>', unsafe_allow_html=True)
            
            antibiotic_data['Aminoglycoside'] = st.radio(
                "Aminoglycoside",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
                key="amino"
            )
            st.markdown('<div class="drug-examples">Gentamicin, Amikacin, Tobramycin</div>', unsafe_allow_html=True)
        
        # Auto-calculated Antibiotic Risk Score
        antibiotic_risk_score = sum(antibiotic_data.values())
        st.markdown(f'<div class="auto-calc">📊 Antibiotic Risk Score: <strong>{antibiotic_risk_score}</strong> (auto-calculated)</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Validation warnings
        warnings = validate_inputs(patient_data, antibiotic_data)
        for warning in warnings:
            st.markdown(f'<div class="validation-warning">{warning}</div>', unsafe_allow_html=True)
        
        # Calculate button
        if st.button("🔬 Calculate CPE Risk", key="calc_btn"):
            # Prepare data for model
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
                'Antibiotic_Risk': antibiotic_risk_score
            }
            
            input_df = pd.DataFrame([model_input])
            input_df = input_df[features]
            
            try:
                probability = model.predict_proba(input_df)[0, 1]
                st.session_state.probability = probability
                st.session_state.show_result = True
                st.session_state.antibiotic_breakdown = antibiotic_data
                st.session_state.antibiotic_total = antibiotic_risk_score
                st.success("✅ Risk assessment completed!")
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
            
            # Results box
            st.markdown(f'''
            <div class="{risk_class}">
                <div class="results-box">
                    <div class="results-title">🎯 CPE Risk Assessment</div>
                    <div class="results-content">
                        <strong>{risk_text}</strong><br>
                        Risk Probability: <strong>{probability*100:.1f}%</strong><br><br>
                        {recommendation}
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Model performance
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('<div class="info-title">📊 Model Performance</div>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="info-content">
                <strong>Sensitivity:</strong> 62% (detects 62% of CPE carriers)<br>
                <strong>ROC-AUC:</strong> 0.71 (good discrimination)<br>
                <strong>Model Threshold:</strong> 45%<br>
                <strong>Validation:</strong> 2,923 ICU patients (2023)<br>
                <strong>CPE Prevalence:</strong> 8-11% in ICU patients
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # About this calculator
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('<div class="info-title">🏥 About This Calculator</div>', unsafe_allow_html=True)
            st.markdown('''
            <div class="info-content">
                <strong>Purpose:</strong> Predicts CPE colonization risk within 48 hours of ICU admission<br><br>
                
                <strong>Algorithm:</strong> Logistic Regression with SMOTE<br>
                <strong>Training Data:</strong> 1,992 ICU admissions (2022)<br>
                <strong>Validation Data:</strong> 2,923 ICU admissions (2023)<br>
                <strong>Institution:</strong> Hallym University Sacred Heart Hospital<br><br>
                
                <strong>Key Predictors:</strong><br>
                • Central venous catheter use<br>
                • Nasogastric tube placement<br>
                • Prior antibiotic exposure<br>
                • Long-term care facility admission<br>
                • Hospital days before ICU<br><br>
                
                <em>This tool is intended for clinical decision support and should be interpreted by qualified healthcare providers.</em>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
