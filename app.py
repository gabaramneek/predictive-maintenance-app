import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('machine_failure_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Predictive Maintenance System")

st.write("Enter machine operating parameters")

# Inputs
machine_type = st.selectbox(
    "Machine Type",
    [0, 1, 2]
)

air_temp = st.number_input("Air Temperature (K)")

process_temp = st.number_input("Process Temperature (K)")

rpm = st.number_input("Rotational Speed (RPM)")

torque = st.number_input("Torque (Nm)")

tool_wear = st.number_input("Tool Wear (min)")

# Feature engineering
temp_difference = process_temp - air_temp

power = torque * rpm

# Prepare input
input_data = np.array([
    [
        machine_type,
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear,
        temp_difference,
        power
    ]
])

# Scale input
input_data = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data)

prediction_proba = model.predict_proba(input_data)

# Button
if st.button("Predict Failure"):

    if prediction[0] == 1:
        st.error("Machine Failure Predicted")

    else:
        st.success("No Machine Failure Predicted")

    st.write(
        "Failure Probability:",
        round(prediction_proba[0][1] * 100, 2),
        "%"
    )