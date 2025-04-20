import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models
dnn_model = load_model("dnn_model.h5")
autoencoder = load_model("autoencoder_model.h5",compile=False)
lstm_model = load_model("lstm_model.h5")

# Load scaler
scaler = joblib.load("scaler.pkl")  # Optional if used

st.title("🔍 Customer Churn Prediction App")

st.sidebar.header("Select Model")
model_choice = st.sidebar.radio("Choose a model:", ("DNN", "Autoencoder", "LSTM"))

st.sidebar.write("---")

# Input fields (customize these to match your dataset features)
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

# Convert inputs to model format
input_data = np.array([[1 if gender == "Male" else 0, senior, tenure, monthly_charges, total_charges]])

# Scale if needed
input_scaled = scaler.transform(input_data)  # Remove if you didn't use scaler

# Predict
if st.button("Predict Churn"):
    if model_choice == "DNN":
        pred = dnn_model.predict(input_scaled)[0][0]
        st.success("Prediction: Churn" if pred > 0.5 else "Prediction: No Churn")
        st.write(f"Confidence: {pred:.2f}")

    elif model_choice == "Autoencoder":
        recon = autoencoder.predict(input_scaled)
        error = np.mean(np.square(input_scaled - recon))
        threshold = 0.01  # Use threshold from training
        st.success("Prediction: Churn" if error > threshold else "Prediction: No Churn")
        st.write(f"Reconstruction Error: {error:.5f}")

    elif model_choice == "LSTM":
        lstm_input = np.reshape(input_scaled, (1, 1, input_scaled.shape[1]))
        pred = lstm_model.predict(lstm_input)[0][0]
        st.success("Prediction: Churn" if pred > 0.5 else "Prediction: No Churn")
        st.write(f"Confidence: {pred:.2f}")
