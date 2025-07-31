import streamlit as st
import numpy as np
import pickle

with open("model_top3.pkl", "rb") as file:
    model = pickle.load(file)
st.title("Churn Prediction Model")
st.title("UTS P1 - 2025")
st.subheader("Authors: Murray Atkin, Ibrahim Kassem, Bradley Moore, Preeti Sowrab")
tenure = st.number_input("Tenure")
monthly_charges = st.number_input("Monthly Charges")
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Example: encode categorical input if needed
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
input_data = np.array([[tenure, monthly_charges, contract_map[contract_type]]])
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")

if st.button("Go to Churn Analysis 📊"):
    st.switch_page("churn_analysis.py")  

