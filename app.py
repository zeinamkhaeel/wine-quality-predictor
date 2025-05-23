import streamlit as st
import numpy as np
import joblib

# === Background image using Imgur ===
st.markdown(
    """
    <style>
    .stApp {
        background: none;
        margin-top: -5rem;
        position: relative;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background-image: url("https://i.imgur.com/1c5bD5B.jpeg");
        background-size: cover;
        background-position: center;
        filter: blur(8px) brightness(0.4);
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = joblib.load('wine_model.pkl')

st.title("🍷 Wine Quality Predictor")
st.write("Input the wine characteristics to predict if it's good quality.")

alcohol = st.slider("Alcohol", 8.0, 15.0, step=0.1)
sulphates = st.slider("Sulphates", 0.3, 2.0, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, step=0.01)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, step=0.01)
density = st.slider("Density", 0.9900, 1.0040, step=0.0001)
chlorides = st.slider("Chlorides", 0.01, 0.6, step=0.01)

input_data = np.array([[alcohol, sulphates, citric_acid, volatile_acidity, density, chlorides]])
prediction = model.predict(input_data)[0]

st.markdown("### Prediction:")
if prediction == 1:
    st.success("✅ Good Quality Wine")
else:
    st.error("⚠️ Not Good Quality Wine")
