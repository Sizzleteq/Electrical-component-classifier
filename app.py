import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("arx_system_model.pkl")

# -----------------------------
# Label mapping
# -----------------------------
label_map = {
    0: "Resistor",
    1: "Capacitor",
    2: "Inductor",
    3: "Diode",
    4: "Transistor"
}

# -----------------------------
# Feature columns (MUST match training)
# -----------------------------
feature_columns = [
    "resistance_ohm",
    "capacitance_f",
    "inductance_h",
    "forward_voltage_v",
    "current_gain",
    "power_rating_w",
    "max_voltage_v"
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Electrical Component Classifier", layout="centered")

st.title("🔌 Electrical Component Classifier")
st.write("Enter the electrical parameters to predict the component type.")

# Input fields
resistance = st.number_input("Resistance (Ohms)", value=0.0)
capacitance = st.number_input("Capacitance (Farads)", value=0.0, format="%.10f")
inductance = st.number_input("Inductance (Henrys)", value=0.0)
forward_voltage = st.number_input("Forward Voltage (V)", value=0.0)
current_gain = st.number_input("Current Gain", value=0.0)
power_rating = st.number_input("Power Rating (W)", value=0.0)
max_voltage = st.number_input("Max Voltage (V)", value=0.0)

# Predict button
if st.button("Predict Component"):
    input_df = pd.DataFrame([[
        resistance,
        capacitance,
        inductance,
        forward_voltage,
        current_gain,
        power_rating,
        max_voltage
    ]], columns=feature_columns)

    prediction = pipeline.predict(input_df)
    component = label_map[int(prediction[0])]

    st.success(f"✅ Predicted Component: *{component}*")