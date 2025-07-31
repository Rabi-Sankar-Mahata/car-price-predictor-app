import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ✅ Load preprocessor and model
preprocessor = joblib.load("preprocessor.joblib")
model = joblib.load("regressor.joblib")

st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction App")
st.write("Enter the details below to predict the car selling price.")

# ✅ Input Fields
name = st.selectbox("Car Brand", ["Maruti", "Hyundai", "Honda", "Toyota", "Ford", "BMW", "Mercedes"])
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000)
seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)
max_power = st.number_input("Max Power (BHP)", min_value=10.0, max_value=500.0, value=80.0)
mileage = st.number_input("Mileage (KMPL)", min_value=1.0, max_value=50.0, value=18.0)
engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, value=1200)

# ✅ Create Input DataFrame
input_data = pd.DataFrame([{
    "name": name,
    "fuel": fuel,
    "owner": owner,
    "year": year,
    "km_driven": km_driven,
    "seats": seats,
    "max_power (in bph)": max_power,
    "Mileage(KMPL)": mileage,
    "Engine (CC)": engine
}])

# ✅ Predict Button
if st.button("Predict Price"):
    try:
        # Transform input
        X_transformed = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(X_transformed)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
