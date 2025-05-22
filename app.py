import streamlit as st
import numpy as np
import joblib

# === CSS: background, overlay, blur, and layout ===
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images5.alphacoders.com/443/443997.jpg");
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
        background-color: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }

    .glass-box {
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        margin: 3rem auto;
        max-width: 700px;
        width: 90%;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
    }

    .result-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    .glass-box h1, .glass-box p, label {
        color: #f9f6f2 !important;
    }

    .stSlider > div > div {
        background-color: #ffffff22 !important;
    }

    </style>
""", unsafe_allow_html=True)

# === Start layout ===
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

# === Title ===
st.markdown(
    "<h1 style='text-align: center; color: #ffcc70;'>🍷 Wine Quality Predictor</h1>",
    unsafe_allow_html=True
)

st.write("Adjust the wine properties below and click Predict Quality.")

# === Load model ===
model = joblib.load("wine_model.pkl")

# === Sliders ===
alcohol = st.slider("Alcohol", 8.0, 15.0)
sulphates = st.slider("Sulphates", 0.3, 2.0)
citric_acid = st.slider("Citric Acid", 0.0, 1.0)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6)
density = st.slider("Density", 0.9900, 1.0040)
chlorides = st.slider("Chlorides", 0.01, 0.6)

# === Predict on button click ===
if st.button("🔍 Predict Quality"):
    input_data = np.array([[alcohol, sulphates, citric_acid, volatile_acidity, density, chlorides]])
    prediction = model.predict(input_data)[0]

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    if prediction == 1:
        st.success("✅ This wine is likely GOOD quality.")
    else:
        st.error("⚠️ This wine is likely NOT good quality.")
    st.markdown('</div>', unsafe_allow_html=True)

# === About ===
st.markdown("### 📌 About this App")
st.markdown("""
This app uses a machine learning model trained on the [Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)  
to predict if red wine is likely to be good quality.

Created with ❤️ by **Zeina Mkhaeel**  
🔗 [GitHub](https://github.com/zeinamkhaeel)
""")

# === End glass box ===
st.markdown('</div>', unsafe_allow_html=True)
