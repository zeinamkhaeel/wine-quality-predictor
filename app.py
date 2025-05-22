import streamlit as st
import numpy as np
import joblib

# === CSS for background and layout ===
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
        background: rgba(255, 255, 255, 0.9);
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
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    .glass-box h1, .glass-box h2, .glass-box h3, .glass-box p,
    .result-box p, .stMarkdown, label {
        color: #222222 !important;
        font-weight: 600;
    }

    .stSlider > div > div {
        background-color: #00000030 !important;
    }

    .footer {
        margin-top: 3rem;
        font-size: 0.9rem;
        text-align: center;
        color: #444444;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Begin container ===
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

st.title("🍷 Wine Quality Predictor")
st.write("Input the wine characteristics below and click 'Predict Quality' to check if the wine is good.")

# Load model
model = joblib.load("wine_model.pkl")

# Sliders
alcohol = st.slider("Alcohol", 8.0, 15.0, step=0.1)
sulphates = st.slider("Sulphates", 0.3, 2.0, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, step=0.01)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, step=0.01)
density = st.slider("Density", 0.9900, 1.0040, step=0.0001)
chlorides = st.slider("Chlorides", 0.01, 0.6, step=0.01)

# Predict only when button is clicked
if st.button("🔍 Predict Quality"):
    input_data = np.array([[alcohol, sulphates, citric_acid, volatile_acidity, density, chlorides]])
    prediction = model.predict(input_data)[0]

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    if prediction == 1:
        st.success("✅ This wine is likely **GOOD** quality.")
    else:
        st.error("⚠️ This wine is likely **NOT good** quality.")
    st.markdown('</div>', unsafe_allow_html=True)

# About Section
st.markdown("### 📌 About this App")
st.markdown("""
This simple machine learning app predicts whether a red wine is likely to be good quality based on its chemical properties.

It was trained on the [Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) using a Random Forest classifier.

**Created with ❤️ by Zeina Mkhaeel**  
🔗 [GitHub](https://github.com/zeinamkhaeel)
""")

st.markdown('</div>', unsafe_allow_html=True)
