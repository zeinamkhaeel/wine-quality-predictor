import streamlit as st
import numpy as np
import joblib

# === CSS for background and glass container ===
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
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        margin: 3rem auto;
        width: 90%;
        max-width: 700px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    .result-box {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    .glass-box h1, .glass-box p, .result-box p {
        color: #000000;
        font-weight: 600;
    }

    .stSlider > div > div {
        background-color: #00000030 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Start of Glass Box ===
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

st.title("🍷 Wine Quality Predictor")
st.write("Input the wine characteristics to predict if it's good quality.")

# Load your model
model = joblib.load("wine_model.pkl")

# Sliders
alcohol = st.slider("Alcohol", 8.0, 15.0, step=0.1)
sulphates = st.slider("Sulphates", 0.3, 2.0, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, step=0.01)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, step=0.01)
density = st.slider("Density", 0.9900, 1.0040, step=0.0001)
chlorides = st.slider("Chlorides", 0.01, 0.6, step=0.01)

# Predict
input_data = np.array([[alcohol, sulphates, citric_acid, volatile_acidity, density, chlorides]])
prediction = model.predict(input_data)[0]

# === Result Box ===
st.markdown('<div class="result-box">', unsafe_allow_html=True)
if prediction == 1:
    st.success("✅ This wine is likely GOOD quality.")
else:
    st.error("⚠️ This wine is likely NOT good quality.")
st.markdown('</div>', unsafe_allow_html=True)

# === End of Glass Box ===
st.markdown('</div>', unsafe_allow_html=True)
