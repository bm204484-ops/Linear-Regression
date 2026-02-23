import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("Diabetes Progression Prediction")

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

st.subheader("Model Performance Metrics")
col1, col2 = st.columns(2)
col1.metric("MSE", f"{mean_squared_error(y_test, pred):.2f}")
col2.metric("R2 Score", f"{r2_score(y_test, pred):.2f}")

st.divider()

st.subheader("Predict Diabetes Progression")
st.write("Enter feature values (standardized) to get a prediction:")

age = st.number_input("Age", value=0.0)
bmi = st.number_input("BMI", value=0.0)
bp = st.number_input("Blood Pressure", value=0.0)
s1 = st.number_input("S1 (Blood Serum)", value=0.0)

if st.button("Predict"):
    input_data = np.zeros((1, 10))
    input_data[0, 0] = age
    input_data[0, 2] = bmi
    input_data[0, 3] = bp
    input_data[0, 4] = s1
    
    prediction = model.predict(input_data)
    st.info(f"Predicted Disease Progression: {prediction[0]:.2f}")
