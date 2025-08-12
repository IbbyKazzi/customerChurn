import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
import plotly.express as px
import os



def run():  
    path = r"/mount/src/customerchurn/models/model_3_v2.pkl"
    #st.write(os.getcwd())
    with open(path, "rb") as file:
        model = pickle.load(file)
    st.subheader("Churn Prediction Model")
    st.subheader("UTS P1 - 2025")
    st.write("Authors: Murray Atkin, Ibrahim Kassem, Bradley Moore, Preeti Sowrab")

    #st.subheader("💡 Prediction calculator")
    st.markdown("<h4 style='font-size:18px;'>💡 Prediction calculator</h3>", unsafe_allow_html=True)
    tenure = st.number_input("Months", min_value=0, step=1, format="%d") #get tenure as a whole number
    monthly_charges = st.number_input("Monthly Charges") #get monthly charges as a decimal number
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    # encode categorical input of contract
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    input_data = np.array([[tenure, monthly_charges, contract_map[contract_type]]])
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")
    import modelsPerformance
    modelsPerformance.run()    
