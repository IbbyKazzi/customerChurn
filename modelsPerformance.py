import pickle
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def run(): 
    path1 = r"/mount/src/customerchurn/models/model_3_v4.pkl"
    path2 = r"/mount/src/customerchurn/models/model_21_v4.pkl"
    #st.write(path2)
    with open(path1, "rb") as f:
        model_t3 = pickle.load(f)
    
    with open(path2, "rb") as f:
        model_all = pickle.load(f)
    
    
    #load the dataset
    import load_dataset
    df_encoded = load_dataset.run()  #this function returnes encoded dataset with 22 features  
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
    st.sidebar.metric(label="ROC AUC Score:", value=f"{auc_score * 100:.0f} %", delta=None, delta_color=delta_color)
