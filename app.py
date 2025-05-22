import streamlit as st
import numpy as np
import joblib

# === CSS Styling ===
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images5.alphacoders.com/443/443997.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: relative;
        margin-top: -5rem;
    }

    .glass-box {
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem auto;
        width: 90%;
        max-width: 700px;
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

    .glass-box h1, .glass-box p, label, .result-box p {
        color: #f9f6f2 !important;
        font-weight: 600;
    }

    /* Improve slider visibility */
    .stSlider > div {
        padding: 6px 0 !important;
    }

    .stSlider > div > div {
        background: #f0f0f0 !important;  /* Lighter track */
        border-radius: 4px;
        height: 4px !important;
    }

    .stSlider input[type=range]::-webkit-slider-thumb {
        background: #ffcc70 !important;  /* Gold thumb */
        border: 2px solid white;
        height: 16px;
        width: 16px;
        border-radius: 50%;
    }

    .stSlider input[type=range]:focus {
        outline: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# === Start Layout ===
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #ffcc70;'>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.write("Adjust the wine properties below and click Predict Quality.")

# Load model
model = joblib.load("wine_model.pkl")

# Input sliders
alcohol = st.slider("Alcohol", 8.0, 15.0, step=0.1)
sulphates = st.slider("Sulphates", 0.3, 2.0, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, step=0.01)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, step=0.01)
density = st.slider("Density", 0.9900, 1.0040, step=0.0001)
chlorides = st.slider("Chlorides", 0.01, 0.6, step=0.01)

# Predict button
if st.button("🔍 Predict Quality"):
    input_data = np.array([[alcohol, sulphates, citric_acid, volatile_acidity, density, chlorides]])
    prediction = model.predict(input_data)[0]

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    if prediction == 1:
        st.success("✅ This wine is likely GOOD quality.")
    else:
        st.error("⚠️ This wine is likely NOT good quality.")
    st.markdown('</div>', unsafe_allow_html=True)

# About
st.markdown("### 📌 About this App")
st.markdown("""
This wine predictor uses a machine learning model to determine the quality of red wine based on several chemical attributes.  
Built with Scikit-learn and Streamlit by **Zeina Mkhaeel**.

🔗 [GitHub](https://github.com/zeinamkhaeel)
""")

st.markdown('</div>', unsafe_allow_html=True)
