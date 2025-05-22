import streamlit as st
import numpy as np
import joblib

# === CSS Styling ===
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://images5.alphacoders.com/443/443997.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: relative;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.3);  /* Lighten background */
        z-index: -1;
    }

    .glass-box {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 2rem;
        margin: 3rem auto;
        max-width: 700px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }

    .glass-box h1, label, p {
        color: #222 !important;
        font-weight: 600;
    }

    .stSlider > div > div {
        background-color: #ffffff55 !important;
    }
    </style>
""", unsafe_allow_html=True)

# === App layout ===
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #c45f1a;'>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.write("Adjust wine properties and click to predict quality.")

# Load model
model = joblib.load("wine_model.pkl")

# Inputs
alcohol = st.slider("Alcohol", 8.0, 15.0)
sulphates = st.slider("Sulphates", 0.3, 2.0)
citric_acid = st.slider("Citric Acid", 0.0, 1.0)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6)
density = st.slider("Density", 0.9900, 1.0040)
chlorides = st.slider("Chlorides", 0.01, 0.6)

# Prediction
if st.button("🔍 Predict Quality"):
    input_data = np.array([[alcohol, sulphates, citric_acid, volatile_acidity, density, chlorides]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("✅ This wine is likely GOOD quality.")
    else:
        st.error("⚠️ This wine is likely NOT good quality.")

# About
st.markdown("### 📌 About this App")
st.markdown("""
This app uses a machine learning model to predict the quality of red wine based on its chemical properties.

Built by **Zeina Mkhaeel** · 🔗 [GitHub](https://github.com/zeinamkhaeel)
""")

# Close glass container
st.markdown('</div>', unsafe_allow_html=True)
