import streamlit as st
import joblib
import numpy as np

# === Custom styling ===
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

    /* Predict button styling */
    .stButton > button {
        font-size: 18px !important;
        padding: 0.75em 1.5em;
        border-radius: 8px;
        background-color: #ffcc70;
        color: black;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #e6a940;
        color: black;
    }

    /* Add visual spacing below result */
    .result-spacer {
        margin-top: 30px;
        margin-bottom: 40px;
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

# === Title and instructions ===
st.title("🍷 Wine Quality Predictor")
st.write("Adjust the wine characteristics and press **Predict Quality**.")

# === Input Sliders ===
alcohol = st.slider("Alcohol", 8.0, 15.0, step=0.1)
sulphates = st.slider("Sulphates", 0.3, 2.0, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, step=0.01)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, step=0.01)
density = st.slider("Density", 0.9900, 1.0040, step=0.0001)
chlorides = st.slider("Chlorides", 0.01, 0.6, step=0.01)

# === Predict Button and Output ===
if st.button("🔍 Predict Quality"):
    input_data = np.array([[alcohol, sulphates, citric_acid, volatile_acidity, density, chlorides]])
    prediction = model.predict(input_data)[0]

    st.markdown("### Prediction Result:")
    if prediction == 1:
        st.success("✅ This wine is likely GOOD quality.")
    else:
        st.error("⚠️ This wine is likely NOT good quality.")

    # Visual spacing before About section
    st.markdown('<div class="result-spacer"></div>', unsafe_allow_html=True)

# === About the App ===
with st.expander("📌 About this App"):
    st.markdown("""
    This wine quality prediction tool uses a trained machine learning model to estimate wine quality 
    based on chemical features such as acidity, alcohol, and sulphates.

    **Author**: Zeina Mkhaeel  
    🔗 [GitHub](https://github.com/zeinamkhaeel)
    """)
