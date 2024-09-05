import streamlit as st
import numpy as np
import pickle

# Load the machine learning model
model_filename = 'diabetes.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def make_prediction(features):
    prediction = model.predict([features])
    return prediction

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton button { background-color: #4CAF50; color: white; font-size: 18px; border-radius: 10px; }
    .title { color: #4CAF50; font-size: 36px; text-align: center; }
    .subtitle { color: #3c3c3c; font-size: 24px; text-align: center; margin-bottom: 40px; }
    .stNumberInput label { font-size: 18px; color: #3c3c3c; }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle of the app
st.markdown('<p class="title">Diabetes Prediction Application</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter your details to predict the likelihood of diabetes</p>', unsafe_allow_html=True)

# Layout for input fields
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=140, step=1)
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)

    with col2:
        Glucose = st.number_input("Glucose", min_value=0, max_value=200, step=1)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
        Age = st.number_input("Age", min_value=1, max_value=120, step=1)

# Create an array of inputs
inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

# Add some spacing
st.write("")

# Button to make predictions
if st.button("Predict"):
    prediction = make_prediction(inputs)
    
    # Display the result with styling
    if prediction[0] == 1:
        st.markdown('<p style="color: red; font-size: 24px; text-align: center;">The predicted outcome is: Positive for Diabetes</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: green; font-size: 24px; text-align: center;">The predicted outcome is: Negative for Diabetes</p>', unsafe_allow_html=True)

# Footer or additional information
st.markdown("---")
st.markdown('<p style="text-align: center;">Made with ❤️ by Gokul</p>', unsafe_allow_html=True)
