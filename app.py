import streamlit as st
import joblib
import numpy as np

# === Background Image ===
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-image: url("https://i.imgur.com/1c5bD5B.jpeg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    h1, label, .stMarkdown {
        color: white !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Load the model safely ===
try:
    model = joblib.load("wine_model.pkl")
except:
    st.error("⚠️ Could not load model file. Please check 'wine_model.pkl'.")
    st.stop()

# === Title ===
st.title("🍷 Wine Quality Predictor")
st.write("Adjust the wine characteristics and press **Predict Quality**:")

# === Input Sliders ===
alcohol = st.slider("Alcohol", 8.0, 15.0, step=0.1)
sulphates = st.slider("Sulphates", 0.3, 2.0, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, step=0.01)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, step=0.01)
density = st.slider("Density", 0.9900, 1.0040, step=0.0001)
chlorides = st.slider("Chlorides", 0.01, 0.6, step=0.01)

# === Prediction Button ===
if st.button("Predict Quality"):
    input_data = np.array([[alcohol, sulphates, citric_acid, volatile_acidity, density, chlorides]])
    prediction = model.predict(input_data)[0]

    st.markdown("### Prediction Result:")
    if prediction == 1:
        st.success("✅ This wine is likely GOOD quality.")
    else:
        st.error("⚠️ This wine is likely NOT good quality.")

# === About Section ===
with st.expander("📌 About this App"):
    st.markdown("""
    This wine quality prediction tool uses a trained machine learning model to estimate wine quality 
    based on chemical features such as acidity, alcohol, and sulphates.

    **Author**: Zeina Mkhaeel  
    🔗 [GitHub](https://github.com/zeinamkhaeel)
    """)
