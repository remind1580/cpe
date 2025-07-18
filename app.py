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

# Paper-optimized styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Clean paper background */
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        color: #000000;
        padding: 2rem 0;
    }
    
    /* Large main title for paper */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: -0.02em;
    }
    
    /* Remove all card borders */
    .main-container {
        background: transparent;
        border: none;
        padding: 0;
        margin: 0;
    }
    
    /* Large section headers - same size as title */
    .large-section-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #000000;
        margin: 2rem 0 1.5rem 0;
        letter-spacing: -0.02em;
    }
    
    /* Medium subsection headers */
    .medium-subsection-header {
        font-size: 2rem;
        font-weight: 600;
        color: #000000;
        margin: 2rem 0 1rem 0;
    }
    
    /* Large input labels */
    .stSelectbox > label, .stNumberInput > label {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: #000000 !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Clean input styling - no boxes */
    .stSelectbox > div > div {
        background: transparent;
        border: 2px solid #000000;
        border-radius: 0;
        font-size: 1.25rem;
        padding: 0.75rem;
    }
    
    .stNumberInput > div > div > input {
        background: transparent;
        border: 2px solid #000000;
        border-radius: 0;
        font-size: 1.25rem;
        padding: 0.75rem;
        color: #000000;
    }
    
    /* Large primary button */
    .stButton > button {
        background: #000000;
        color: #FFFFFF;
        border: 2px solid #000000;
        border-radius: 0;
        padding: 1.5rem 4rem;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #333333;
        border-color: #333333;
    }
    
    /* Clean gauge container */
    .gauge-container {
        background: #F8F9FA;
        border: 2px solid #000000;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    
    /* Large metrics */
    .metric-large {
        font-size: 1.75rem;
        font-weight: 600;
        color: #000000;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Risk levels - large and clear */
    .risk-high { 
        color: #DC2626; 
        font-weight: 700;
        font-size: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .risk-medium { 
        color: #D97706; 
        font-weight: 700;
        font-size: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .risk-low { 
        color: #059669; 
        font-weight: 700;
        font-size: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Model info box */
    .model-info-box {
        background: #F1F5F9;
        border: 2px solid #000000;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .model-info-box h3 {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #000000;
    }
    
    .model-info-box ul {
        margin: 0;
        padding-left: 2rem;
    }
    
    .model-info-box li {
        font-size: 1.25rem;
        line-height: 1.8;
        color: #000000;
        margin: 0.5rem 0;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    /* Remove default streamlit padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
</style>
""", unsafe_allow_html=True)

# Large title
st.markdown('<h1 class="main-title">CPE Risk Predictor</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('cpe_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_paper_gauge(probability, threshold=0.45):
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
        title = {
            'text': f"<b style='color:{color}; font-size:28px;'>{risk_level}</b>",
            'font': {'size': 24}
        },
        number = {
            'font': {'size': 60, 'color': color, 'family': 'Inter'}, 
            'suffix': '%',
            'valueformat': '.1f'
        },
        gauge = {
            'axis': {
                'range': [None, 100], 
                'tickcolor': "#000000", 
                'tickfont': {'size': 16, 'color': '#000000', 'family': 'Inter'},
                'tickwidth': 2
            },
            'bar': {'color': color, 'thickness': 0.6},
            'bgcolor': "#FFFFFF",
            'borderwidth': 3,
            'bordercolor': "#000000",
            'steps': [
                {'range': [0, 30], 'color': "#E5F7E5"},
                {'range': [30, 45], 'color': "#FFF3E0"},
                {'range': [45, 100], 'color': "#FFEBEE"}
            ],
            'threshold': {
                'line': {'color': "#000000", 'width': 4},
                'thickness': 0.8,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#000000",
        font_family="Inter",
        height=400,
        margin=dict(l=20, r=20, t=80, b=20)
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
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        # Large section header
        st.markdown('<div class="large-section-header">📋 Patient Clinical Information</div>', unsafe_allow_html=True)
        
        patient_data = {}
        
        # Hospital & Care Settings
        st.markdown('<div class="medium-subsection-header">🏥 Hospital & Care Settings</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            patient_data['Hospital days before ICU admission'] = st.number_input(
                "Hospital days before ICU admission", 
                min_value=0, max_value=100, value=5
            )
        with col_b:
            patient_data['Admission to long-term care facility'] = st.selectbox(
                "Admission from long-term care facility", 
                [0, 1], 
                format_func=lambda x: "Yes" if x else "No"
            )
        
        # Medical Conditions  
        st.markdown('<div class="medium-subsection-header">🩺 Medical Conditions</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="medium-subsection-header">🔌 Invasive Devices</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="medium-subsection-header">💊 Antibiotic Exposure</div>', unsafe_allow_html=True)
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
                min_value=0, max_value=10, value=2
            )
        
        # Calculate button
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Calculate CPE Risk"):
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
        # Large section header
        st.markdown('<div class="large-section-header">📊 Risk Assessment</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'show_result') and st.session_state.show_result:
            probability = st.session_state.probability
            
            # Large gauge chart
            st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
            fig = create_paper_gauge(probability)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Large metrics
            st.markdown(f'<div class="metric-large">Risk Probability: <strong>{probability*100:.1f}%</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-large">Model Threshold: <strong>45.0%</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-large">ROC-AUC: <strong>0.774</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-large">Sensitivity: <strong>72.5%</strong></div>', unsafe_allow_html=True)
            
            # Clinical recommendation
            if probability >= 0.45:
                st.markdown('<div class="risk-high">⚠️ HIGH RISK<br>Consider CPE isolation precautions</div>', unsafe_allow_html=True)
            elif probability >= 0.3:
                st.markdown('<div class="risk-medium">⚡ MODERATE RISK<br>Enhanced monitoring recommended</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">✅ LOW RISK<br>Standard care protocols</div>', unsafe_allow_html=True)
        else:
            # Model information box
            st.markdown('<div class="model-info-box">', unsafe_allow_html=True)
            st.markdown('<h3>🤖 Model Information</h3>', unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li><strong>Algorithm:</strong> Logistic Regression</li>
                <li><strong>Features:</strong> 12 clinical variables</li>
                <li><strong>ROC-AUC:</strong> 0.774</li>
                <li><strong>Sensitivity:</strong> 72.5%</li>
                <li><strong>Specificity:</strong> 68.9%</li>
                <li><strong>Training Data:</strong> 3,932 patients</li>
                <li><strong>Validation:</strong> Temporal (2022→2023)</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
