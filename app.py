import streamlit as st
import numpy as np
import pickle
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
import streamlit as st

selected = option_menu(
    menu_title=None,
    options=["Home", "Analysis", "Cust Service", "Retention"],
    icons=["house", "bar-chart", "person-lines-fill","shield-check" ],
    orientation="horizontal"
)

if selected == "Home":
    st.title("🏠 Home Page")
    run_home()
elif selected == "Analysis":
    st.title("📊 Analysis Page")
elif selected == "Cust Service":
    st.title("👤 Customer Service Page")
elif selected == "Retention":
    st.title("🛡️ Retention Page")
    run_retention()


def run_home():    
    with open("model_top3.pkl", "rb") as file:
        model = pickle.load(file)
    st.title("Churn Prediction Model")
    st.title("UTS P1 - 2025")
    st.subheader("Authors: Murray Atkin, Ibrahim Kassem, Bradley Moore, Preeti Sowrab")
    tenure = st.number_input("Tenure", min_value=0, step=1, format="%d") #get tenure as a whole number
    monthly_charges = st.number_input("Monthly Charges") #get monthly charges as a decimal number
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    # encode categorical input of contract
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    input_data = np.array([[tenure, monthly_charges, contract_map[contract_type]]])
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")
    
    #if st.button("Go to Churn Analysis 📊"):
    #    st.switch_page("churn_analysis.py")  
    
    github_url = "https://customerchurn-utsp1-analysis.streamlit.app/"
    
    st.markdown(
        f"""
        <a href="{github_url}" target="_blank">
            <button style="background-color:#1f77b4; color:white; padding:10px 24px; font-size:16px; border:none; border-radius:5px;">
                🚀 Go to Churn Analysis 📊
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )






