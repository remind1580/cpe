import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="CPE Risk Predictor",
    layout="wide",
    page_icon="🧪"
)

# -----------------------
# Custom CSS (Figma-like)
# -----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{
  --primary:#1E40AF;      /* deep blue */
  --secondary:#2563EB;    /* bright blue */
  --success:#10B981;      /* green */
  --danger:#DC2626;       /* red */
  --muted:#6B7280;        /* gray */
  --bg:#F7F8FA;           /* light bg */
  --card:#FFFFFF;
  --radius:14px;
}
html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
  background: var(--bg);
}
h1,h2,h3 { color:#111827; font-weight:700; }
.small { color:var(--muted); font-size:0.92rem; }

/* Cards */
.section-card{
  background:var(--card);
  border-radius:var(--radius);
  padding:20px 20px 16px 20px;
  box-shadow:0 6px 20px rgba(17,24,39,0.06);
  border:1px solid #EEF2F7;
}

/* Inputs grouping */
.block-label{
  font-weight:600; margin-bottom:8px; display:block;
}

/* Radio group -> pill style */
div[role="radiogroup"] > label {
  border:1px solid #E5E7EB;
  background:#F3F6FB;
  padding:8px 14px; margin-right:8px; margin-bottom:6px;
  border-radius:12px; cursor:pointer;
}
div[role="radiogroup"] > label:hover{ border-color:#CBD5E1; }
div[role="radiogroup"] input:checked + div { background: var(--secondary) !important; color:#fff !important; }

/* Buttons */
.stButton > button{
  background:var(--secondary);
  color:#fff; font-weight:700; border:none;
  padding:10px 16px; border-radius:12px;
  transition: all .15s ease-in-out;
}
.stButton > button:hover{ background:var(--primary); }

/* Result banners */
.result-positive{
  border-left:6px solid var(--danger);
  background:#FEF2F2; color:#991B1B;
  border-radius:12px; padding:14px 16px; font-weight:600;
}
.result-negative{
  border-left:6px solid var(--success);
  background:#ECFDF5; color:#065F46;
  border-radius:12px; padding:14px 16px; font-weight:600;
}

/* Responsive: columns naturally stack on small screens */
@media (max-width: 900px){
  .section-card{ padding:16px; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Title
# -----------------------
st.title("CPE (Carbapenemase-producing Enterobacterales) Risk Predictor")

# -----------------------
# Load Model
# -----------------------
MODEL_PATH = Path("cpe_model.pkl")
if not MODEL_PATH.exists():
    st.error("Model file 'cpe_model.pkl' not found. Place it next to app.py and redeploy.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model_blob = pickle.load(f)

model = model_blob.get("model")
model_features = model_blob.get("features", [])
threshold = float(model_blob.get("threshold", 0.45))
model_info = model_blob.get("model_info", {})

if model is None or not model_features:
    st.error("Model or feature list missing in cpe_model.pkl.")
    st.stop()

# -----------------------
# Helper: feature vector builder
# -----------------------
# We allow flexible alias matching between UI labels and model feature names.
# All matching is done in lowercase.
FEATURE_ALIASES = {
    "hospital days before icu admission": ["hospital days before icu admission", "hospital_days_before_icu_admission", "hospital_days"],
    "admission to long-term care facility": ["admission to long-term care facility", "admission to long-term care facilities", "admission_longtermcare", "long_term_care"],
    "esrd on renal replacement therapy": ["esrd on renal replacement therapy", "esrd", "esrd_renal_replacement"],
    "steroid use": ["steroid use", "steroid", "steroid use within 3 months"],
    "vre": ["vre", "vre colonization", "vre colonization within 6 months"],
    "endoscopy": ["endoscopy", "endoscopy within 1 year"],
    "central venous catheter": ["central venous catheter", "cvc"],
    "nasogastric tube": ["nasogastric tube", "ng tube", "ng_tube"],
    "biliary drain": ["biliary drain", "ptbd", "biliary drainage"],
    "β-lactam/β-lactamase inhibitor": ["β-lactam/β-lactamase inhibitor", "blbli", "beta-lactam/beta-lactamase inhibitor"],
    "cephalosporin": ["cephalosporin"],
    "fluoroquinolone": ["fluoroquinolone"],
    "carbapenem": ["carbapenem"],
    "aminoglycoside": ["aminoglycoside"],
    "antibiotic_risk": ["antibiotic_risk", "antibiotic risk"]
}

def resolve_feature_name(target_lower: str):
    """
    Given a lowercased model feature name, try to resolve a canonical key
    from FEATURE_ALIASES dict. If not found, return the original.
    """
    for canonical, aliases in FEATURE_ALIASES.items():
        for a in aliases:
            if target_lower == a.lower():
                return canonical
    return target_lower

def assemble_input_vector(model_features, ui_values_dict):
    """
    Build the input vector in the exact order of model_features.
    ui_values_dict should contain canonical keys (as defined above).
    Missing features default to 0.
    """
    vector = []
    for f in model_features:
        key = resolve_feature_name(str(f).lower())
        vector.append(ui_values_dict.get(key, 0))
    return np.array(vector, dtype=float).reshape(1, -1)
