import streamlit as st
import joblib
import numpy as np

# === Safe Background Image CSS (NO z-index or ::before) ===
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.imgur.com/1c5bD5B.jpeg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding: 2rem;
    }

    h1, .stMarkdown, label {
        color: #ffffff !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Load model ===
model = joblib.load("wine_model.pkl")

# === Title ===
st.title("🍷 Wine Quality Predictor")
st.write("Input the wine characteristics to predict if it's good quality.")

# === Input sliders ===
alcohol = st.slider("Alcohol", 8.0, 15.0, step=0.1)
sulphates = st.slider("Sulphates", 0.3, 2.0, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, step=0.01)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, step=0.01)
density = st.slider("Density", 0.9900, 1.0040, step=0.0001)
chlorides = st.slider("Chlorides", 0.01, 0.6, step=0.01)

# === Prediction ===
input_data = np.array([[alcohol, sulphates, citric_acid, volatile_acidity, density, chlorides]])
prediction = model.predict(input_data)[0]

st.markdown("### Prediction:")
if prediction == 1:
    st.success("✅ Good Quality Wine")
else:
    st.error("⚠️ Not Good Quality Wine")
