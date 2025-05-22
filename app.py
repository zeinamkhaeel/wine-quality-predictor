import streamlit as st
import numpy as np
import joblib

# --- 1. CSS Styling for Background + Glassmorphism ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images5.alphacoders.com/443/443997.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .glass-box {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem auto;
        width: 90%;
        max-width: 700px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }

    .glass-box h1, .glass-box h2, .glass-box h3, .glass-box p {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 2. Begin Glass Container ---
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

# --- 3. App Content ---
st.title("🍷 Wine Quality Predictor")
st.write("Input wine characteristics below to predict if it's good quality.")

# Load model
model = joblib.load("wine_model.pkl")

# Feature sliders
alcohol = st.slider("Alcohol", 8.0, 15.0, step=0.1)
sulphates = st.slider("Sulphates", 0.3, 2.0, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, step=0.01)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, step=0.01)
density = st.slider("Density", 0.9900, 1.0040, step=0.0001)
chlorides = st.slider("Chlorides", 0.01, 0.6, step=0.01)

# Prediction
input_data = np.array([[alcohol, sulphates, citric_acid, volatile_acidity, density, chlorides]])
prediction = model.predict(input_data)[0]

st.markdown("### Prediction:")
if prediction == 1:
    st.success("✅ This wine is likely GOOD quality.")
else:
    st.error("⚠️ This wine is likely NOT good quality.")

# --- 4. Close Glass Container ---
st.markdown('</div>', unsafe_allow_html=True)
