import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models
dnn_model = load_model("dnn_model.h5")
autoencoder = load_model("autoencoder_model.h5", compile=False)
lstm_model = load_model("lstm_model.h5")

# Load scaler
scaler = joblib.load("scaler.pkl")

st.title("🔍 Telecom Customer Churn Prediction")

st.sidebar.header("Choose Model")
model_choice = st.sidebar.radio("Select model:", ("DNN", "Autoencoder", "LSTM"))

st.sidebar.markdown("---")

# Create input form for 30 features
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (in months)", 0)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", 0.0)
total_charges = st.number_input("Total Charges", 0.0)

# Encode the categorical features (match training encoding)
def binary_encode(val): return 1 if val == "Yes" else 0

# Sample encoding — you must match with how you trained!
input_list = [
    1 if gender == "Male" else 0,
    binary_encode(senior),
    binary_encode(partner),
    binary_encode(dependents),
    tenure,
    binary_encode(phone_service),
    1 if multiple_lines == "Yes" else 0,
    {"No": 0, "DSL": 1, "Fiber optic": 2}[internet_service],
    {"No": 0, "Yes": 1, "No internet service": 2}[online_security],
    {"No": 0, "Yes": 1, "No internet service": 2}[online_backup],
    {"No": 0, "Yes": 1, "No internet service": 2}[device_protection],
    {"No": 0, "Yes": 1, "No internet service": 2}[tech_support],
    {"No": 0, "Yes": 1, "No internet service": 2}[streaming_tv],
    {"No": 0, "Yes": 1, "No internet service": 2}[streaming_movies],
    {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
    binary_encode(paperless_billing),
    {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[payment_method],
    monthly_charges,
    total_charges
]

# Add dummy 30-feature padding (for demo)
# Normally you'll use all exact input features here
while len(input_list) < 30:
    input_list.append(0.0)

input_array = np.array([input_list])
input_scaled = scaler.transform(input_array)

print("Input shape to scaler:", input_array.shape)
print("Scaler was trained on:", scaler.mean_.shape)

# Check expected input size
expected_features = scaler.mean_.shape[0]  # or scaler.scale_.shape[0]

if input_array.shape[1] != expected_features:
    st.error(f"Expected {expected_features} features, but got {input_array.shape[1]}")
    st.stop()

# Check for feature mismatch
expected_input = dnn_model.input_shape[-1]
if input_scaled.shape[1] != expected_input:
    st.error(f"Model expects {expected_input} features, got {input_scaled.shape[1]}")
    st.stop()

# Predict
if st.button("Predict Churn"):
    if model_choice == "DNN":
        pred = dnn_model.predict(input_scaled)[0][0]
        st.success("🔴 Churn" if pred > 0.5 else "🟢 No Churn")
        st.write(f"Confidence: {pred:.2f}")

    elif model_choice == "Autoencoder":
        recon = autoencoder.predict(input_scaled)
        error = np.mean(np.square(input_scaled - recon))
        threshold = 0.01  # Adjust based on your training
        st.success("🔴 Churn" if error > threshold else "🟢 No Churn")
        st.write(f"Reconstruction Error: {error:.5f}")

    elif model_choice == "LSTM":
        lstm_input = np.reshape(input_scaled, (1, 1, input_scaled.shape[1]))
        pred = lstm_model.predict(lstm_input)[0][0]
        st.success("🔴 Churn" if pred > 0.5 else "🟢 No Churn")
        st.write(f"Confidence: {pred:.2f}")
