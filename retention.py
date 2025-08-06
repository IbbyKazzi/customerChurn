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

def run():
    #get the prediction model    
    path = r"/mount/src/customerchurn/models/model_20_v3.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    #import the dataset
    X = pd.read_csv("encoded-dataset.csv")
    #explainer = shap.Explainer(model, X)
    #shap_values = explainer(X)
    
    # Load your dataset to extract customer ids
    df = pd.read_csv("Customer-Churn-dataset.csv")
    df = df[df['Churn'] == 'No']
    
    #print model's feature order
    #st.write(model.feature_names_in_)
    
    # tenure group in 3 categories, New - Loyal - Long-term
    def tenure_group(tenure):
        if tenure <= 12:
            return 'New'
        elif 12 < tenure <= 24:
            return 'Loyal'
        else:
            return 'Long-term'

    df['tenure_group'] = df['tenure'].apply(tenure_group)
    
    # Recreate MonthlyCharges_Tenure if it was a product
    df["MonthlyCharges_Tenure"] = df["MonthlyCharges"] * df["tenure"]
    
    #remove unwated features
    cols_to_drop = ["customerID", "Churn"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    #st.write("🔎 Columns currently in df:")
    #st.write(df.columns.tolist())
    
    def align_features(df, model):
        return df[model.feature_names_in_]
    
    df_aligned = align_features(df, model)
    
    #encode the dataset
    df_encoded = df_aligned.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        # Exclude 'TotalCharges' for now as it seems to have non-numeric values that need handling
        if col not in ['customerID', 'TotalCharges']:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    
    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    
    # Drop rows with NaN values created by the conversion
    df_encoded.dropna(inplace=True) 
    
    # Predict probabilities
    churn_probs = model.predict_proba(df_encoded)[:, 1]   
    
    
    # Add the prediction back into your DataFrame
    df_encoded["churn_probability"] = churn_probs
    
    #Set the risk thresholds via streamlit slider for a dynamic input
    st.sidebar.header("Set Risk Thresholds")
    
    high_threshold = st.sidebar.slider("High Risk Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    medium_threshold = st.sidebar.slider("Medium Risk Threshold", min_value=0.0, max_value=high_threshold, value=0.4, step=0.01)
    
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
        percent = count / len(df)
        st.write(f"{tier}: {count} customers")
        st.progress(percent)  
