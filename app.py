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

# Clean clinical styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Clean white background */
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        color: #1F2937;
    }
    
    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1F2937;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: -0.025em;
    }
    
    /* Clean card styling */
    .clinical-card {
        background: #FFFFFF;
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    
    /* Subsection headers */
    .subsection-header {
        font-size: 1.125rem;
        font-weight: 600;
        color: #4B5563;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Input labels - larger and clearer */
    .stSelectbox > label, .stNumberInput > label {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Clean input styling */
    .stSelectbox > div > div {
        background: #F9FAFB;
        border: 2px solid #D1D5DB;
        border-radius: 8px;
        font-size: 1rem;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .stNumberInput > div > div > input {
        background: #F9FAFB;
        border: 2px solid #D1D5DB;
        border-radius: 8px;
        font-size: 1rem;
        padding: 0.75rem;
    }
    
    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 1rem 3rem;
        font-size: 1.125rem;
        font-weight: 600;
        letter-spacing: 0.025em;
        transition: all 0.2s;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8, #1E40AF);
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Results section */
    .results-card {
        background: #F8FAFC;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    /* Metrics styling */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #E5E7EB;
        font-size: 1.125rem;
    }
    
    .metric-label {
        font-weight: 500;
        color: #374151;
    }
    
    .metric-value {
        font-weight: 600;
        color: #1F2937;
        font-size: 1.25rem;
    }
    
    /* Risk level colors */
    .risk-high { 
        color: #DC2626; 
        font-weight: 700;
        font-size: 1.25rem;
    }
    .risk-medium { 
        color: #D97706; 
        font-weight: 700;
        font-size: 1.25rem;
    }
    .risk-low { 
        color: #059669; 
        font-weight: 700;
        font-size: 1.25rem;
    }
    
    /* Model info */
    .model-info {
        background: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .model-info ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .model-info li {
        font-size: 1rem;
        line-height: 1.6;
        color: #4B5563;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🏥 CPE Risk Predictor</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('cpe_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_clean_gauge(probability, threshold=0.45):
    if probability >= threshold:
        risk_level = "HIGH RISK"
        color = "#DC2626"
    elif probability >= 0.3:
        risk_level = "MODERATE RISK"
        color = "#D97706"
    else:
        risk_level = "LOW RISK"
        color = "#059669"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b style='color:{color}; font-size:24px;'>{risk_level}</b>"},
        number = {'font': {'size': 48, 'color': color}, 'suffix': '%'},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': "#4B5563", 'tickfont': {'size': 14}},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': "#F8FAFC",
            'borderwidth': 2,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 30], 'color': "#DCFCE7"},
                {'range': [30, 45], 'color': "#FEF3C7"},
                {'range': [45, 100], 'color': "#FEE2E2"}
            ],
            'threshold': {
                'line': {'color': "#3B82F6", 'width': 3},
                'thickness': 0.8,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#374151",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

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
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📋 Patient Clinical Information</div>', unsafe_allow_html=True)
        
        patient_data = {}
        
        # Hospital & Care Settings
        st.markdown('<div class="subsection-header">🏥 Hospital & Care Settings</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            patient_data['Hospital days before ICU admission'] = st.number_input(
                "Hospital days before ICU admission", 
                min_value=0, max_value=100, value=5,
                help="Number of days patient was hospitalized before ICU admission"
            )
        with col_b:
            patient_data['Admission to long-term care facility'] = st.selectbox(
                "Admission from long-term care facility", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No",
                help="Recent admission from long-term care facility"
            )
        
        # Medical Conditions  
        st.markdown('<div class="subsection-header">🩺 Medical Conditions</div>', unsafe_allow_html=True)
        col_c, col_d = st.columns(2)
        with col_c:
            patient_data['ESRD on renal replacement therapy'] = st.selectbox(
                "ESRD on renal replacement therapy", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No"
            )
            patient_data['VRE'] = st.selectbox(
                "VRE colonization", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No"
            )
        with col_d:
            patient_data['Steroid use'] = st.selectbox(
                "Steroid use", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No"
            )
            patient_data['Endoscopy'] = st.selectbox(
                "Recent endoscopy", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No"
            )
        
        # Invasive Devices
        st.markdown('<div class="subsection-header">🔌 Invasive Devices</div>', unsafe_allow_html=True)
        col_e, col_f = st.columns(2)
        with col_e:
            patient_data['Central venous catheter'] = st.selectbox(
                "Central venous catheter", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No"
            )
            patient_data['Nasogastric tube'] = st.selectbox(
                "Nasogastric tube", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No"
            )
        with col_f:
            patient_data['Biliary drain'] = st.selectbox(
                "Biliary drain", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No"
            )
        
        # Antibiotic Exposure
        st.markdown('<div class="subsection-header">💊 Antibiotic Exposure</div>', unsafe_allow_html=True)
        col_g, col_h = st.columns(2)
        with col_g:
            patient_data['Carbapenem'] = st.selectbox(
                "Carbapenem use", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No"
            )
            patient_data['Aminoglycoside'] = st.selectbox(
                "Aminoglycoside use", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No"
            )
        with col_h:
            patient_data['Antibiotic_Risk'] = st.number_input(
                "Antibiotic Risk Score", 
                min_value=0, max_value=10, value=2,
                help="Cumulative antibiotic exposure risk score"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate button
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("Calculate CPE Risk", use_container_width=True):
                input_df = pd.DataFrame([patient_data])
                input_df = input_df[features]
                
                try:
                    probability = model.predict_proba(input_df)[0, 1]
                    st.session_state.probability = probability
                    st.session_state.show_result = True
                    st.success("✅ Risk assessment completed!")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    with col2:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Risk Assessment</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'show_result') and st.session_state.show_result:
            probability = st.session_state.probability
            
            # Gauge chart
            fig = create_clean_gauge(probability)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            st.markdown('<div class="results-card">', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="metric-row">
                <span class="metric-label">Risk Probability</span>
                <span class="metric-value">{probability*100:.1f}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Model Threshold</span>
                <span class="metric-value">45.0%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">ROC-AUC</span>
                <span class="metric-value">0.774</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Sensitivity</span>
                <span class="metric-value">72.5%</span>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clinical recommendation
            st.markdown("**Clinical Recommendation**")
            if probability >= 0.45:
                st.markdown('<p class="risk-high">⚠️ HIGH RISK - Consider CPE isolation precautions</p>', unsafe_allow_html=True)
            elif probability >= 0.3:
                st.markdown('<p class="risk-medium">⚡ MODERATE RISK - Enhanced monitoring recommended</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="risk-low">✅ LOW RISK - Standard care protocols</p>', unsafe_allow_html=True)
        else:
            st.info("👆 Enter patient information and click 'Calculate CPE Risk'")
            
            # Model information
            st.markdown('<div class="model-info">', unsafe_allow_html=True)
            st.markdown("**🤖 Model Information**")
            st.markdown("""
            - **Algorithm:** Logistic Regression
            - **Features:** 12 clinical variables
            - **ROC-AUC:** 0.774
            - **Sensitivity:** 72.5%
            - **Specificity:** 68.9%
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
