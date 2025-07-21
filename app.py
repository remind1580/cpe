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

# Enhanced CSS with better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400;1,600&family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    /* Root variables for consistent design */
    :root {
        --primary-blue: #2563EB;
        --dark-blue: #1E40AF;
        --light-blue: #DBEAFE;
        --success-green: #10B981;
        --warning-orange: #F59E0B;
        --danger-red: #DC2626;
        --text-primary: #111827;
        --text-secondary: #6B7280;
        --bg-primary: #F9FAFB;
        --bg-white: #FFFFFF;
        --border-color: #E5E7EB;
    }
    
    /* Global styles */
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    /* Main title with elegant serif font */
    .main-title {
        font-family: 'Crimson Text', serif;
        font-size: 3.5rem;
        font-weight: 600;
        color: var(--dark-blue);
        text-align: center;
        margin-bottom: 3rem;
        font-style: italic;
        letter-spacing: -0.02em;
        line-height: 1.2;
        text-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Section boxes with subtle shadows */
    .section-box {
        background: var(--bg-white);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.06);
        transition: all 0.2s ease;
    }
    
    .section-box:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06);
    }
    
    /* Section headers with elegant italic style (restored) */
    .section-header {
        font-family: 'Lato', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c5282;
        font-style: italic;
        text-transform: capitalize;
        letter-spacing: 0.3px;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    /* #1 Fix: Force larger Yes/No button text with multiple approaches */
    /* Remove default Streamlit radio styling */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 1.5rem !important;
        display: flex !important;
        flex-direction: row !important;
        margin: 1rem 0 2rem 0 !important;
    }
    
    /* Style radio button containers */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background-color: var(--bg-primary) !important;
        border: 3px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1.5rem 3rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        flex: 0 0 auto !important;
        min-width: 140px !important;
        text-align: center !important;
        position: relative !important;
    }
    
    /* Force text size for radio options - same size as labels */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label span,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label p {
        font-size: 1.5rem !important;  /* Same as label size */
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    /* Selected state */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background-color: var(--primary-blue) !important;
        border-color: var(--primary-blue) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) span,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p {
        color: white !important;
    }
    
    /* Hover state */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        border-color: var(--primary-blue) !important;
        background-color: var(--light-blue) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Hide radio input circles */
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    
    /* #2 Fix: Larger question labels */
    div[data-testid="stRadio"] > label,
    div[data-testid="stNumberInput"] > label {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 1rem !important;
        line-height: 1.4 !important;
        display: block !important;
    }
    
    div[data-testid="stRadio"] > label p,
    div[data-testid="stNumberInput"] > label p {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    
    /* #3 Fix: Remove number input spinner buttons and style */
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
        display: none !important;
    }
    
    input[type="number"] {
        -moz-appearance: textfield !important;
        appearance: textfield !important;
    }
    
    div[data-testid="stNumberInput"] > div > div > input {
        border: 2px solid var(--border-color) !important;
        border-radius: 10px !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.4rem !important;
        background-color: var(--bg-primary) !important;
        font-weight: 500 !important;
        width: 100% !important;
        text-align: center !important;
    }
    
    div[data-testid="stNumberInput"] > div > div > input:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
        outline: none !important;
    }
    
    /* Hide increment/decrement buttons completely */
    div[data-testid="stNumberInput"] > div > div > button {
        display: none !important;
    }
    
    /* Calculate button with modern gradient */
    .stButton > button {
        background: linear-gradient(135deg, var(--success-green), #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1.25rem 3rem !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.025em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25) !important;
        width: 100% !important;
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35) !important;
    }
    
    /* Results section styling */
    .results-box {
        background: linear-gradient(135deg, var(--success-green), #059669);
        color: white;
        border-radius: 16px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
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
    
    .risk-high .results-box { 
        background: linear-gradient(135deg, var(--danger-red), #B91C1C); 
    }
    .risk-medium .results-box { 
        background: linear-gradient(135deg, var(--warning-orange), #D97706); 
    }
    
    /* Info box styling */
    .info-box {
        background: var(--bg-white);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.5rem 0;
        transition: all 0.2s ease;
    }
    
    .info-box:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .info-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-content {
        font-size: 1.05rem;
        color: var(--text-secondary);
        line-height: 1.7;
    }
    
    .info-content strong {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* Auto-calculated score with modern design */
    .auto-calc {
        background: linear-gradient(135deg, #EBF8FF, #DBEAFE);
        border: 2px solid var(--primary-blue);
        border-radius: 12px;
        padding: 1.25rem;
        font-size: 1.25rem;
        color: var(--dark-blue);
        text-align: center;
        margin-top: 1.5rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1);
    }
    
    /* Validation warnings */
    .validation-warning {
        background: #FEF2F2;
        border-left: 4px solid var(--danger-red);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #991B1B;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #D1FAE5 !important;
        color: #065F46 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border-left: 4px solid var(--success-green) !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    /* Hide help tooltips to remove white spaces */
    [data-testid="stTooltipIcon"] {
        display: none !important;
    }
    
    /* Remove extra spacing from streamlit containers */
    .element-container:has(> [data-testid="stTooltipIcon"]) {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-title { font-size: 2.5rem; }
        .section-header { font-size: 1.5rem; }
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            flex-direction: column !important;
            gap: 1rem !important;
        }
    }
    /* Remove white box */
    .info-box:empty {
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }
    /* Remove white section-box */
    .section-box:empty {
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }

</style>

<script>
    // Force style application for Streamlit components
    function applyCustomStyles() {
        // Force radio button text to be larger
        const radioLabels = document.querySelectorAll('div[role="radiogroup"] label span');
        radioLabels.forEach(label => {
            label.style.fontSize = '1.5rem';
            label.style.fontWeight = '700';
        });
        
        // Force widget labels to be larger
        const widgetLabels = document.querySelectorAll('[data-testid="stWidgetLabel"] p');
        widgetLabels.forEach(label => {
            label.style.fontSize = '1.5rem';
            label.style.fontWeight = '600';
        });
    }
    
    // Apply styles on load and mutations
    const observer = new MutationObserver(applyCustomStyles);
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Initial application
    setTimeout(applyCustomStyles, 100);
    setTimeout(applyCustomStyles, 500);
    setTimeout(applyCustomStyles, 1000);
</script>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-title">CPE (Carbapenemase-producing Enterobacterales) Risk Predictor</h1>', unsafe_allow_html=True)

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
        warnings.append("⚠️ Patient on RRT with 0 hospital days - please verify")
    
    antibiotic_count = sum(antibiotic_data.values())
    if antibiotic_count >= 4:
        warnings.append("⚠️ High antibiotic exposure (4+ classes) - increased CPE risk")
    
    if patient_data.get('Central venous catheter') == 1 and patient_data.get('Nasogastric tube') == 1 and patient_data.get('Biliary drain') == 1:
        warnings.append("⚠️ Multiple invasive devices present - consider enhanced monitoring")
    
    return warnings

def main():
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.error("❌ Failed to load prediction model. Please check model file.")
        return
        
    model = model_data['model']
    features = model_data['features']
    
    # Main layout - adjusted column ratio for better space utilization
    col1, col2 = st.columns([2.5, 1], gap="large")
    
    with col1:
        patient_data = {}
        antibiotic_data = {}
        
        # Part A: Healthcare Exposure
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part A: Healthcare Exposure & Admission History</div>', unsafe_allow_html=True)
        
        col_a1, col_a2 = st.columns(2, gap="medium")
        with col_a1:
            patient_data['Hospital days before ICU admission'] = st.number_input(
                "Hospital days before ICU admission : Count days from current admission to ICU transfer", 
                min_value=0, 
                max_value=365, 
                value=0
            )
            
        with col_a2:
            patient_data['Admission to long-term care facility'] = st.radio(
                "Admission from long-term care facility : Nursing home, Long-Term Acute Care, or rehabilitation facility",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="ltc"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Part B: Medical Conditions
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part B: Medical Conditions & Interventions</div>', unsafe_allow_html=True)
        
        col_b1, col_b2 = st.columns(2, gap="medium")
        with col_b1:
            patient_data['ESRD on renal replacement therapy'] = st.radio(
                "ESRD on renal replacement therapy : Dialysis or CRRT required",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="esrd"
            )
            
            patient_data['VRE'] = st.radio(
                "VRE colonization : Previous VRE colonization (any time)",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="vre"
            )
            
        with col_b2:
            patient_data['Steroid use'] = st.radio(
                "Steroid use : Systemic corticosteroids within 3 months",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="steroid"
            )
            
            patient_data['Endoscopy'] = st.radio(
                "Recent endoscopy : Any endoscopic procedure within 30 days",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="endoscopy"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Part C: Invasive Devices
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part C: Invasive Devices (within 48 hours of ICU admission)</div>', unsafe_allow_html=True)
        
        col_c1, col_c2, col_c3 = st.columns(3, gap="medium")
        with col_c1:
            patient_data['Central venous catheter'] = st.radio(
                "Central venous catheter : Subclavian, jugular, femoral line",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="cvc"
            )
            
        with col_c2:
            patient_data['Nasogastric tube'] = st.radio(
                "Nasogastric tube or Orogastric tube placement",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="ngt"
            )
            
        with col_c3:
            patient_data['Biliary drain'] = st.radio(
                "Biliary drain : PTBD, ERCP stent, drainage procedures",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="biliary"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Part D: Antibiotic Exposure
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Part D: Antibiotic Exposure (within 3 months before ICU admission)</div>', unsafe_allow_html=True)
        
        col_d1, col_d2 = st.columns(2, gap="medium")
        with col_d1:
            antibiotic_data['Fluoroquinolone'] = st.radio(
                "Fluoroquinolone : Ciprofloxacin, Levofloxacin, Moxifloxacin",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="fluoro"
            )
            
            antibiotic_data['Cephalosporin'] = st.radio(
                "Cephalosporin : Ceftriaxone, Cefazolin, Ceftazidime, Cefepime",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="ceph"
            )
            
            antibiotic_data['Carbapenem'] = st.radio(
                "Carbapenem : Meropenem, Imipenem, Ertapenem, Doripenem",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="carbap"
            )
            
        with col_d2:
            antibiotic_data['β-lactam/β-lactamase inhibitor'] = st.radio(
                "β-lactam/β-lactamase inhibitor : Piperacillin/Tazobactam, Ampicillin/Sulbactam",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="beta_lactam"
            )
            
            antibiotic_data['Aminoglycoside'] = st.radio(
                "Aminoglycoside : Gentamicin, Amikacin, Tobramycin",
                [0, 1],
                format_func=lambda x: "YES" if x else "NO",
                horizontal=True,
                key="amino"
            )
        
        # Auto-calculated score
        antibiotic_risk_score = sum(antibiotic_data.values())
        st.markdown(f'<div class="auto-calc">📊 Antibiotic Risk Score: <strong>{antibiotic_risk_score}</strong> (auto-calculated)</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Validation warnings
        warnings = validate_inputs(patient_data, antibiotic_data)
        for warning in warnings:
            st.markdown(f'<div class="validation-warning">{warning}</div>', unsafe_allow_html=True)
        
        # Calculate button
        if st.button("🔬 CALCULATE CPE RISK", key="calc_btn", use_container_width=True):
            # Prepare model input
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
                st.success("✅ Risk assessment completed successfully!")
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
    
    with col2:
        # Results section
        if hasattr(st.session_state, 'show_result') and st.session_state.show_result:
            probability = st.session_state.probability
            
            # Risk classification
            if probability >= 0.45:
                risk_class = "risk-high"
                risk_text = "HIGH RISK"
                risk_emoji = "🔴"
                recommendation = "Immediate CPE screening and isolation precautions recommended"
            elif probability >= 0.3:
                risk_class = "risk-medium"
                risk_text = "MODERATE RISK"
                risk_emoji = "🟡"
                recommendation = "Enhanced monitoring and standard precautions"
            else:
                risk_class = "risk-low"
                risk_text = "LOW RISK"
                risk_emoji = "🟢"
                recommendation = "Standard infection control measures"
            
            # Results display
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
            
            # Model metrics
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('<div class="info-title">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="info-content">
                <strong>Sensitivity:</strong> 62.0%<br>
                <strong>Specificity:</strong> 68.5%<br>
                <strong>ROC-AUC:</strong> 0.71<br>
                <strong>PPV:</strong> 17.8%<br>
                <strong>NPV:</strong> 94.2%<br>
                <strong>Threshold:</strong> 45%
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # About section
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('''
            <div class="info-title">🏥 About This Calculator</div>
            <div class="info-content" style="font-size:1.1rem; color:#1F2937; line-height:1.8; font-weight:500">
                **Purpose**  
                Predicts CPE colonization risk within 48 hours of ICU admission.
        
                **Algorithm**  
                Logistic Regression without SMOTE
                
                **Training / Validation Data**  
                Combined dataset of 2022 and 2023 ICU admissions  
                - Total: 4,915 ICU admissions  
                - Split: 80% training, 20% validation
                
                **Key Predictors**  
                • Central venous catheter use  
                • Nasogastric tube placement  
                • Prior antibiotic exposure  
                • Long-term care facility admission  
                • Hospital days before ICU admission  
                
                This tool is intended for clinical decision support and should be interpreted by qualified healthcare providers.
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # About researcher section (#3 addition)
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('<div class="info-title">👨‍⚕️ About the Researcher</div>', unsafe_allow_html=True)
            st.markdown('''
            <div class="info-content">
                <strong>Principal Investigator:</strong> [Name]<br>
                <strong>Institution:</strong> Hallym University Sacred Heart Hospital<br>
                <strong>Department:</strong> Infectious Diseases<br>
                <strong>Research Focus:</strong> Antimicrobial resistance & Healthcare epidemiology<br><br>
                
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clinical guidelines
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown('<div class="info-title">📋 Clinical Guidelines</div>', unsafe_allow_html=True)
            st.markdown('''
            <div class="info-content">
              High Risk (≥45%)
                • Immediate rectal swab screening
                • Contact isolation precautions
                • Cohort nursing if possible
                
               Moderate Risk (30-44%):
                • Consider screening based on local epidemiology
                • Enhanced hand hygiene compliance
                
                <strong>Low Risk (<30%):
                • Standard precautions
                • Routine surveillance per protocol
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
