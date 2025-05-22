import streamlit as st

import streamlit as st

def set_bg_image():
    st.markdown(
        """
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
            width: 100%;
            background-image: url('https://images.unsplash.com/photo-1604917877931-84c43b1a1e5f?auto=format&fit=crop&w=1500&q=80');
            background-size: cover;
            background-position: center;
            opacity: 0.15;
            z-index: -1;
        "></div>
        """,
        unsafe_allow_html=True
    )

set_bg_image()  # <- this must come after the function


import joblib
import numpy as np

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
