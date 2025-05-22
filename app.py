import streamlit as st
import numpy as np
import joblib

# === Enhanced CSS ===
st.markdown(
    """
    <style>
    /* Background blur + darkness */
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
        background-image: url("https://images5.alphacoders.com/443/443997.jpg");
        background-size: cover;
        background-position: center;
        filter: blur(10px) brightness(0.4);
        z-index: -1;
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

    /* Improved slider style */
    div[data-baseweb="slider"] {
        padding: 20px 0;
    }

    div[data-baseweb="slider"] > div {
        background-color: #cccccc !important;
        height: 12px !important;
        border-radius: 6px;
    }

    div[data-baseweb="slider"] [role="slider"] {
        background-color: #ffcc70 !important;
        border: 2px solid #ffffff;
        height: 24px !important;
        width: 24px !important;
        border-radius: 50%;
        box-shadow: 0 0 6px rgba(255, 204, 112, 0.7);
    }

    div[data-baseweb="slider"] span {
        color: #f9f6f2 !important;
        font-weight: bold;
        font-size: 16px;
    }

    .css-1v0mbdj { padding-top: 0rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# === Glass Container Start ===
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

# === Title ===
st.markdown(
    "<h1 style='text-align: center; color: #ffcc70;'>🍷 Wine Quality Predictor</h1>",
    unsafe_allow_html=True
)

st.write("Adjust the chemical properties below and click 'Predict Quality' to see if the wine is likely good.")

# === Load model ===
model = joblib.load("wine_model.pkl")

# === Sliders ===
alcohol = st.slider("Alcohol", 8.0, 15.0, step=0.1)
sulphates = st.slider("Sulphates", 0.3, 2.0, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, step=0.01)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, step=0.01)
density = st.slider("Density", 0.9900, 1.0040, step=0.0001)
chlorides = st.slider("Chlorides", 0.01, 0.6, step=0.01)

# === Predict Button ===
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
This wine predictor uses a machine learning model to determine the quality of red wine based on several chemical attributes.  
Built with Scikit-learn and Streamlit by **Zeina Mkhaeel**.

🔗 [GitHub](https://github.com/zeinamkhaeel)
""")

# === End of Container ===
st.markdown('</div>', unsafe_allow_html=True)
