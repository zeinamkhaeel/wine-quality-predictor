import streamlit as st
import joblib
import numpy as np

import streamlit as st

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
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.6);  /* Light white overlay */
        z-index: -1;
    }

    /* Optional: brighten up sliders and text */
    .block-container {
        color: #000000 !important;       /* Make text black */
    }

    .css-1cpxqw2, .stSlider {
        background-color: #ffffff20 !important;
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
