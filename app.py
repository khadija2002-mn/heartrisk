import streamlit as st
import numpy as np
import tensorflow as tf

# Charger le modèle
model = tf.keras.models.load_model("model.h5")

# Interface utilisateur
st.title("Heart Attack Risk Predictor")

age = st.slider("Age", 20, 80, 50)
chol = st.slider("Cholesterol", 100, 400, 200)
thalach = st.slider("Max Heart Rate", 60, 200, 150)

if st.button("Predict"):
    input_data = np.array([[age, chol, thalach]])  # adapte si besoin
    prediction = model.predict(input_data)[0][0]
    st.success(f"Predicted Risk: {prediction*100:.2f}%")
