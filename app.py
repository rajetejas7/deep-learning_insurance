import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model and scaler
try:
    model = load_model('ann_model.keras')
    scaler = joblib.load('scaler.pkl')
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop() # Stop the app if models cannot be loaded

st.title("Insurance Charge Prediction")

st.write("Enter the details of the individual to predict their insurance charges.")

# Input fields for user
age = st.slider("Age", 18, 65, 30)
bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
children = st.slider("Number of Children", 0, 5, 1)
sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Prepare input data for prediction
if st.button("Predict Charges"):
    # Create a DataFrame for the new data point, ensuring correct columns and order
    data_for_prediction = pd.DataFrame([[age, bmi, children,
                                         1 if sex == 'male' else 0,
                                         1 if smoker == 'yes' else 0,
                                         1 if region == 'northwest' else 0,
                                         1 if region == 'southeast' else 0,
                                         1 if region == 'southwest' else 0]],
                                       columns=['age', 'bmi', 'children', 'sex_male', 'smoker_yes',
                                                'region_northwest', 'region_southeast', 'region_southwest'])

    # Scale the numerical features using the pre-fitted scaler
    scaled_input = scaler.transform(data_for_prediction)

    # Make prediction
    prediction = model.predict(scaled_input)[0][0]

    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")
