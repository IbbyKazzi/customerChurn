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
import streamlit as st
import pandas as pd
import os
from settings import MODEL_PATH_T3, MODEL_PATH_T21, DATA_PATH

def run():
    #get the prediction model    
    
    with open(MODEL_PATH_T21, "rb") as f:
        model = pickle.load(f)
    #import the dataset
    #X = pd.read_csv("encoded-dataset.csv")
    #explainer = shap.Explainer(model, X)
    #shap_values = explainer(X)
    
    #load the dataset
    import load_dataset
    df_encoded = load_dataset.run()  #this function returnes encoded dataset with 22 features  
    df_encoded = df_encoded.drop(['Churn'], axis=1) 
    
    #def align_features(df, model):
    #    return df[model.feature_names_in_]
    
    #df_aligned = align_features(df, model)
    
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
        percent = count / len(df_encoded)
        st.write(f"{tier}: {count} customers")
        st.progress(percent)     
    
    
    
    # Upload a new dataset
    st.write("")    
    uploaded_file = st.file_uploader("📂 Upload a new dataset", type=["csv"])
    if "overwrite_done" not in st.session_state:
        st.session_state["overwrite_done"] = False
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("CSV file loaded successfully!")
        st.dataframe(df.head())
    
        # Step 2: Confirm overwrite
        if os.path.exists(DATA_PATH) and not st.session_state.overwrite_done:
            st.warning("⚠️ A file already exists at the save location.")
            if st.button("✅ Overwrite existing file"):
                df.to_csv(DATA_PATH, index=False)
                #st.session_state.overwrite_done = True
                st.success(f"File overwritten and saved to: {DATA_PATH}")
                st.rerun()
        elif not os.path.exists(DATA_PATH):
            if st.button("💾 Save file"):
                df.to_csv(DATA_PATH, index=False)
                #st.session_state.overwrite_done = True
                st.success(f"File saved to: {DATA_PATH}")
                st.rerun()
        else:            
            st.info("Please upload a CSV file to proceed.")

    
