import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the machine learning model
model_filename = 'diabetes.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Function to make predictions and plot results
def make_prediction(features):
    prediction = model.predict([features])
    return prediction

# Title of the app
st.title("Diabetes Prediction Application")

# Input fields for the user
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
Glucose = st.number_input("Glucose", min_value=0, max_value=200, step=1)
BloodPressure = st.number_input("BloodPressure", min_value=0, max_value=140, step=1)
SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=100, step=1)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, step=0.01)
Age = st.number_input("Age", min_value=1, max_value=120, step=1)

# Create an array of inputs
inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

# Button to make predictions
if st.button("Predict"):
    prediction = make_prediction(inputs)
    st.write(f"The predicted outcome is: {prediction[0]}")


