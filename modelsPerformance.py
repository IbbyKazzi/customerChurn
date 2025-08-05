import pickle
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def run(): 
    path1 = r"/mount/src/customerchurn/models/model_3_v3.pkl"
    path2 = r"/mount/src/customerchurn/models/model_20_v3.pkl"
    #st.write(path2)
    with open(path1, "rb") as f:
        model_t3 = pickle.load(f)
    
    with open(path2, "rb") as f:
        model_all = pickle.load(f)
    
    #get the dataset
    df = pd.read_csv("Customer-Churn-dataset.csv")
    # tenure group in 3 categories, New - Loyal - Long-term
    def tenure_group(tenure):
        if tenure <= 12:
            return 'New'
        elif 12 < tenure <= 24:
            return 'Loyal'
        else:
            return 'Long-term'
    
    df['tenure_group'] = df['tenure'].apply(tenure_group)

    # Convert column to numeric (in case it's still object type) and fill in missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill NaNs with median
    median_value = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
    
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
    df_encoded.columns = df_encoded.columns.str.strip()
    df_encoded.drop('customerID', axis=1, inplace=True)
    X_All = df_encoded.drop(['Churn'], axis=1)
    
    top_features = ['tenure', 'MonthlyCharges', 'Contract']
    X_top3 = df_encoded[top_features]    
    y = df_encoded['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X_All, y, test_size=0.2, random_state=42)
    
    # Get predicted probabilities
    y_probs = model_all.predict_proba(X_test)[:, 1]
    
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
    
    #st.subheader("📈 Model's ROC Curve Performance:")
    #st.pyplot(fig)
    delta_color = "normal"
    if auc_score >= 0.85:
        delta_color = "inverse"
    elif auc_score < 0.70:
        delta_color = "off"

    st.sidebar.header("📦 Modle Version: 3")
    st.sidebar.metric(label="ROC AUC Score:", value=f"{auc_score * 100:.2f}%", delta=None, delta_color=delta_color)
