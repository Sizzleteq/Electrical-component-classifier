import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("component_classifier.pkl")
scaler = joblib.load("scaler.pkl")

# Label mapping
label_map = {
    0: "Resistor",
    1: "Capacitor",
    2: "Inductor",
    3: "Diode",
    4: "Transistor"
}

st.set_page_config(page_title="Electrical Component Classifier", layout="centered")

st.title("⚡ Electrical Component Classifier")
st.write("Enter the measured electrical values to predict the component type.")

st.divider()

# Input fields
resistance = st.number_input("Resistance (Ohms)", min_value=0.0, value=0.0)
capacitance = st.number_input("Capacitance (Farads)", min_value=0.0, value=0.0, format="%.10f")
inductance = st.number_input("Inductance (Henrys)", min_value=0.0, value=0.0, format="%.10f")
forward_voltage = st.number_input("Forward Voltage (Volts)", min_value=0.0, value=0.0)
current_gain = st.number_input("Current Gain (hFE)", min_value=0.0, value=0.0)
power_rating = st.number_input("Power Rating (Watts)", min_value=0.0, value=0.0)
max_voltage = st.number_input("Max Voltage (Volts)", min_value=0.0, value=0.0)

st.divider()

# Predict button
if st.button("🔍 Predict Component"):
    input_data = np.array([[
        resistance,
        capacitance,
        inductance,
        forward_voltage,
        current_gain,
        power_rating,
        max_voltage
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    confidence = np.max(model.predict_proba(input_scaled)) * 100

    st.success(f"**Predicted Component:** {label_map[prediction]}")
    st.info(f"**Confidence:** {confidence:.2f}%")

st.caption("ML-powered electrical component classification")

