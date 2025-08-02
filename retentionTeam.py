import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#get the prediction model
with open("model_all.pkl", "rb") as f:
    model = pickle.load(f)
#import the dataset
X = pd.read_csv("encoded-dataset.csv")
#explainer = shap.Explainer(model, X)
#shap_values = explainer(X)

# Load your dataset to extract customer ids
df = pd.read_csv("Customer-Churn-dataset.csv")
df = df[df['Churn'] == 'No']
feature_cols = df.columns.tolist()
st.write("📊 Selected Features for Modeling:")
st.write(feature_cols)

#remove unwated features
df = df.drop(columns=["Churn", "customerID", "tenure"], errors="ignore")

# Recreate MonthlyCharges_Tenure if it was a product
df["MonthlyCharges_Tenure"] = df["MonthlyCharges"] * df["tenure"]

# tenure group in 3 categories, New - Loyal - Long-term
def tenure_group(tenure):
    if tenure <= 12:
        return 'New'
    elif 12 < tenure <= 24:
        return 'Loyal'
    else:
        return 'Long-term'

df['tenure_group'] = df['tenure'].apply(tenure_group)

df["churn_probability"] = model.predict_proba(df[feature_cols])[:, 1]

#set risk tires and generat tags
def categorize_risk(prob):
    if prob >= 0.80:
        return "High Risk 🚨"
    elif prob >= 0.50:
        return "Medium Risk ⚠️"
    else:
        return "Low Risk ✅"
        
df["risk_category"] = df["churn_probability"].apply(categorize_risk)

#visualize in streamlit
import plotly.express as px

risk_counts = df["risk_category"].value_counts().reset_index()
fig = px.pie(risk_counts, names="index", values="risk_category", title="Churn Risk Distribution")
st.plotly_chart(fig)

