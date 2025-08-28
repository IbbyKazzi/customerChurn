import pickle
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from settings import MODEL_PATH_T3, MODEL_PATH_T21, DATA_PATH
import json
from datetime import datetime
import os


def run_daily():   
    
    with open("models/model_21_V5.pkl", "rb") as f:
        model_all = pickle.load(f)    
    
    #load the dataset
    import load_dataset
    df_encoded = load_dataset.run()  #this function returnes encoded dataset with 22 features  
    X_All = df_encoded.drop(['Churn'], axis=1)      
      
    y = df_encoded['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X_All, y, test_size=0.2, random_state=42)
    
    # Get predicted probabilities
    y_probs = model_all.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    
    # Compute AUC score
    auc_score = roc_auc_score(y_test, y_probs)
    model_name = type(model_all).__name__
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": MODEL_PATH_T21,
        "model_type": model_name,
        "module": type(model_all).__module__,
        "version": "V5",
        "auc": auc_score   
    }

    json_path = "models/model_daily_results.json"

    try:
        #Load existing data if file exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []
    
        #Append new results
        if isinstance(existing_data, list):
            existing_data.append(results)
        else:
            st.write("❌ Unexpected JSON structure. Expected a list.")
            existing_data = [results]
    
        #Write back to file
        with open(json_path, "w") as f:
            json.dump(existing_data, f, indent=4)
    
        st.write("✅ JSON appended successfully.")
    
    except Exception as e:
        st.write("❌ Error appending to JSON:", e)




    
    #fig, ax = plt.subplots()
    #ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    #ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    #ax.set_xlabel("False Positive Rate")
    #ax.set_ylabel("True Positive Rate")
    #ax.set_title("ROC Curve")
    #ax.legend()
    
    #st.subheader("📈 Model's ROC Curve Performance:")
    #st.pyplot(fig)
    delta_color = "normal"
    if auc_score >= 0.85:
        delta_color = "inverse"
    elif auc_score < 0.70:
        delta_color = "off"

    #versionNo = MODEL_PATH_T21
    #st.sidebar.header("📦 Current Modle Version: "+ versionNo)
    #st.sidebar.metric(label="ROC AUC Score:", value=f"{auc_score * 100:.0f} %", delta=None, delta_color=delta_color)

if __name__ == "__main__":
    run_daily()

