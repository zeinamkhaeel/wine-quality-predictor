# 🍷 Wine Quality Predictor

This is a machine learning-powered web app that predicts whether red wine is of **good quality** based on several chemical attributes. Built using **Python**, **Scikit-learn**, and **Streamlit**, it provides an interactive and visually appealing user experience.


---

## 🚀 Features

- Predicts wine quality instantly as users adjust input sliders
- Beautiful background with transparent styling
- Easy-to-use web interface built with Streamlit
- Model trained on public wine quality dataset
- Hosted live with Streamlit Cloud

---

## 📊 Input Features

The following chemical properties are used for prediction:

- **Alcohol**
- **Sulphates**
- **Citric Acid**
- **Volatile Acidity**
- **Density**
- **Chlorides**

---

## 🧠 Machine Learning

- Model: `RandomForestClassifier` or similar (customize based on what you used)
- Framework: `scikit-learn`
- Trained on: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

---

## 🌐 Live Demo

🔗 [Click here to try the app](https://wine-quality-predictor-ktkrok8u3kvfvuncg6znfs.streamlit.app/)

---

## 📁 Project Structure

```bash
├── app.py              # Streamlit app code
├── wine_model.pkl      # Trained ML model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
