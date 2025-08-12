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
from settings import MODEL_PATH_T3, MODEL_PATH_T21, DATA_PATH
import load_dataset
import uploadNewDataset

def run():
    #get the prediction model 
    with open(MODEL_PATH_T21, "rb") as f:
        model = pickle.load(f)   
    
    #load encoded dataset    
    df_encoded = load_dataset.run()  #this function returnes encoded dataset with 22 features 
    df_encoded = df_encoded[df_encoded["Churn"] == 0].copy() # we only interested in those who didn't churn
    df_encoded = df_encoded.drop(['Churn'], axis=1)   

    #load original dataset
    df = pd.read_csv(DATA_PATH)
    df_filtered = df[df['Churn'] == 'No']
    df.rename(columns={"tenure": "Months"}, inplace=True)     
    
    # Predict probabilities
    churn_probs = model.predict_proba(df_encoded)[:, 1]       
    
    # Add the prediction back into your DataFrame
    df_encoded["churn_probability"] = churn_probs
    #Add it to the original dataset
    df["churn_probability"] = churn_probs
    
    #Set the risk thresholds via streamlit slider for a dynamic input
    st.sidebar.header("Set Risk Thresholds")
    
    high_threshold = st.sidebar.slider("High Risk Threshold", min_value=0.4, max_value=0.8, value=0.5, step=0.01)
    medium_threshold = st.sidebar.slider("Medium Risk Threshold", min_value=0.2, max_value=high_threshold, value=0.3, step=0.01)
    
    #set risk tires and generat tags
    def categorize_risk(prob):
        if prob >= high_threshold:
            return "High Risk 🚨"
        elif prob >= medium_threshold:
            return "Medium Risk ⚠️"
        else:
            return "Low Risk ✅"
            
    df_encoded["risk_category"] = df_encoded["churn_probability"].apply(categorize_risk)
    
    #visualize in streamlit
    import plotly.express as px
    
    risk_counts = df_encoded["risk_category"].value_counts().reset_index()
    fig = px.pie(risk_counts, names="risk_category", values="count", title="Churn Risk Distribution")
    st.plotly_chart(fig)
    
    risk_counts = df_encoded["risk_category"].value_counts()
    
    st.subheader("Risk Tier Distribution")
    
    for tier in ["High Risk 🚨", "Medium Risk ⚠️", "Low Risk ✅"]:
        count = risk_counts.get(tier, 0)
        percent = count / len(df_encoded)
        st.write(f"{tier}: {count} customers")
        st.progress(percent) 

    #Filter customers by thier tier and allow to export data as .csv to share with the retention team
    st.subheader("View Customers by Risk Tier")
    selected_tier = st.selectbox("Choose a risk category", ["High Risk 🚨", "Medium Risk ⚠️", "Low Risk ✅"])
    filtered_df = df[df["risk_category"] == selected_tier]
    
    st.dataframe(filtered_df)
    
    # Export data using a button
    


    #uploadNewDataset.run(True)
