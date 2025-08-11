
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="CPE Risk Predictor", layout="wide", page_icon="🧪")

st.title("CPE (Carbapenemase-producing Enterobacterales) Risk Predictor")

# Load model
MODEL_PATH = Path("cpe_model.pkl")
if not MODEL_PATH.exists():
    st.error("Model file 'cpe_model.pkl' not found.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model_blob = pickle.load(f)

model = model_blob.get("model")
model_features = model_blob.get("features", [])
threshold = float(model_blob.get("threshold", 0.45))

if model is None or not model_features:
    st.error("Model or feature list missing.")
    st.stop()

# User input
st.header("Input Features")
hospital_days = st.number_input("Hospital days before ICU admission", min_value=0, value=0)

# Convert input to vector
input_vector = np.array([[hospital_days if 'hospital days before icu admission' in f.lower() else 0
                          for f in model_features]])

# Prediction
if st.button("Predict"):
    prob = model.predict_proba(input_vector)[0][1]
    if prob >= threshold:
        st.markdown(f"**High risk** ({prob:.2%})", unsafe_allow_html=True)
    else:
        st.markdown(f"**Low risk** ({prob:.2%})", unsafe_allow_html=True)
