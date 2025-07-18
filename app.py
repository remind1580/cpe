
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="CPE Risk Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Replit-inspired styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #21262D 100%);
        font-family: 'Inter', sans-serif;
        color: #F0F6FC;
    }
    
    .main-header {
        background: linear-gradient(90deg, #00D4FF 0%, #5865F2 50%, #7C3AED 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        animation: gradientShift 3s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .prediction-card {
        background: rgba(28, 33, 40, 0.8);
        border: 1px solid rgba(68, 76, 86, 0.5);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        border-color: rgba(0, 212, 255, 0.6);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #00D4FF, #7C3AED);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.3);
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.5);
        filter: brightness(1.1);
    }
    
    .stSelectbox > div > div > div {
        background-color: rgba(28, 33, 40, 0.9);
        border: 1px solid #444C56;
        border-radius: 10px;
        color: #F0F6FC;
    }
    
    .stNumberInput > div > div > input {
        background-color: rgba(28, 33, 40, 0.9);
        border: 1px solid #444C56;
        border-radius: 10px;
        color: #F0F6FC;
    }
    
    .risk-high { color: #F85149; font-weight: bold; text-shadow: 0 0 10px rgba(248, 81, 73, 0.5); }
    .risk-medium { color: #FFA657; font-weight: bold; text-shadow: 0 0 10px rgba(255, 166, 87, 0.5); }
    .risk-low { color: #56D364; font-weight: bold; text-shadow: 0 0 10px rgba(86, 211, 100, 0.5); }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧬 CPE Risk Predictor</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('cpe_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_gauge_chart(probability, threshold=0.45):
    if probability >= threshold:
        risk_level = "HIGH RISK"
        color = "#F85149"
    elif probability >= 0.3:
        risk_level = "MODERATE RISK"
        color = "#FFA657"
    else:
        risk_level = "LOW RISK"
        color = "#56D364"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b style='color:{color}; font-size:24px;'>{risk_level}</b>"},
        number = {'font': {'size': 48, 'color': color}, 'suffix': '%'},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': "rgba(240, 246, 252, 0.8)"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(68, 76, 86, 0.8)",
            'steps': [
                {'range': [0, 30], 'color': "rgba(86, 211, 100, 0.1)"},
                {'range': [30, 45], 'color': "rgba(255, 166, 87, 0.1)"},
                {'range': [45, 100], 'color': "rgba(248, 81, 73, 0.1)"}
            ],
            'threshold': {
                'line': {'color': "#00D4FF", 'width': 4},
                'thickness': 0.8,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="rgba(240, 246, 252, 0.9)",
        height=350
    )
    return fig

def main():
    model_data = load_model()
    
    if model_data is None:
        st.error("Failed to load model")
        return
        
    model = model_data['model']
    features = model_data['features']
    
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Patient Clinical Information")
        
        patient_data = {}
        
        with st.expander("🏥 Hospital & Care Settings", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                patient_data['Hospital days before ICU admission'] = st.number_input(
                    "Hospital days before ICU", min_value=0, max_value=100, value=5
                )
            with col_b:
                patient_data['Admission to long-term care facility'] = st.selectbox(
                    "Long-term care facility", [0, 1], format_func=lambda x: "Yes" if x else "No"
                )
        
        with st.expander("🩺 Medical Conditions", expanded=True):
            col_c, col_d = st.columns(2)
            with col_c:
                patient_data['ESRD on renal replacement therapy'] = st.selectbox(
                    "ESRD on RRT", [0, 1], format_func=lambda x: "Yes" if x else "No"
                )
                patient_data['VRE'] = st.selectbox(
                    "VRE colonization", [0, 1], format_func=lambda x: "Yes" if x else "No"
                )
            with col_d:
                patient_data['Steroid use'] = st.selectbox(
                    "Steroid use", [0, 1], format_func=lambda x: "Yes" if x else "No"
                )
                patient_data['Endoscopy'] = st.selectbox(
                    "Recent endoscopy", [0, 1], format_func=lambda x: "Yes" if x else "No"
                )
        
        with st.expander("🔌 Invasive Devices", expanded=True):
            col_e, col_f = st.columns(2)
            with col_e:
                patient_data['Central venous catheter'] = st.selectbox(
                    "Central venous catheter", [0, 1], format_func=lambda x: "Yes" if x else "No"
                )
                patient_data['Nasogastric tube'] = st.selectbox(
                    "Nasogastric tube", [0, 1], format_func=lambda x: "Yes" if x else "No"
                )
            with col_f:
                patient_data['Biliary drain'] = st.selectbox(
                    "Biliary drain", [0, 1], format_func=lambda x: "Yes" if x else "No"
                )
        
        with st.expander("💊 Antibiotic Exposure", expanded=True):
            col_g, col_h = st.columns(2)
            with col_g:
                patient_data['Carbapenem'] = st.selectbox(
                    "Carbapenem use", [0, 1], format_func=lambda x: "Yes" if x else "No"
                )
                patient_data['Aminoglycoside'] = st.selectbox(
                    "Aminoglycoside use", [0, 1], format_func=lambda x: "Yes" if x else "No"
                )
            with col_h:
                patient_data['Antibiotic_Risk'] = st.number_input(
                    "Antibiotic Risk Score", min_value=0, max_value=10, value=2
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔮 Calculate CPE Risk", use_container_width=True):
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
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Risk Assessment Results")
        
        if hasattr(st.session_state, 'show_result') and st.session_state.show_result:
            probability = st.session_state.probability
            
            fig = create_gauge_chart(probability)
            st.plotly_chart(fig, use_container_width=True)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Risk Probability", f"{probability*100:.1f}%")
                st.metric("Model Threshold", "45.0%")
            with col_m2:
                st.metric("ROC-AUC", "0.774")
                st.metric("Sensitivity", "72.5%")
            
            st.markdown("### 🩺 Clinical Recommendation")
            if probability >= 0.45:
                st.markdown('<p class="risk-high">⚠️ HIGH RISK - Consider CPE isolation precautions</p>', unsafe_allow_html=True)
            elif probability >= 0.3:
                st.markdown('<p class="risk-medium">⚡ MODERATE RISK - Enhanced monitoring recommended</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="risk-low">✅ LOW RISK - Standard care protocols</p>', unsafe_allow_html=True)
        else:
            st.info("👆 Enter patient information and click 'Calculate CPE Risk'")
            st.markdown("### 🤖 Model Information")
            st.markdown("- **Algorithm:** Logistic Regression\n- **Features:** 12 clinical variables\n- **ROC-AUC:** 0.774")
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="text-align: center; color: rgba(240, 246, 252, 0.6);">🧬 CPE Risk Predictor | ML Model for Healthcare</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
