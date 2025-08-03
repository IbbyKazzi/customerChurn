import pickle
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def run():    
    with open("model_top3.pkl", "rb") as f:
        model_t3 = pickle.load(f)
    
    with open("model_all.pkl", "rb") as f:
        model_all = pickle.load(f)
    
    #get the dataset
    df_encoded = pd.read_csv("Customer-Churn-dataset.csv")
    # Encode categorical variables
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        # Exclude 'TotalCharges' for now as it seems to have non-numeric values that need handling
        if col not in ['customerID', 'TotalCharges']:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    
    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    
    # Drop rows with NaN values created by the conversion
    df_encoded.dropna(inplace=True)
    X = df_encoded.drop(['Churn', 'customerID'], axis=1)
    y = df_encoded['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get predicted probabilities
    y_probs = model_t3.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    
    # Compute AUC score
    auc_score = roc_auc_score(y_test, y_probs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    
    st.subheader("📊 ROC Curve Performance", key="per_sh1")
    st.pyplot(fig, key="per_pt1")
    st.metric(label="ROC AUC Score", value=f"{auc_score:.2f}", key="per_mt1")
