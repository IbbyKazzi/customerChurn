import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import openai
import matplotlib.pyplot as plt
#Load dataset module
import load_dataset
from matplotlib import rcParams
import pickle
import shap
from settings import MODEL_PATH_T21
from sklearn.pipeline import Pipeline
import pickle
import pandas as pd
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
import plotly.express as px
import os
from settings import MODEL_PATH_T3, MODEL_PATH_T21, DATA_PATH
import load_dataset
import uploadNewDataset
import io


# Churn rate overtime
def churnRateTimeline(churned_df, df):
    # --- Input: Current Month Churn Rate ---
    # this is the current churn rate from our dataset
    current_churn = len(churned_df) * 100 / len(df)
    
    # --- Generate illustrative data ---
    months = pd.date_range(end=pd.Timestamp.today(), periods=7, freq='M').strftime('%b %Y').tolist()
    
    # Simulate churn rates: trending down, but previous month has a spike
    base_rate = current_churn
    churn_rates = [base_rate + i*2.5 for i in reversed(range(5))]  # downward trend
    churn_rates += [base_rate + 2.0]  # previous month spike
    churn_rates += [base_rate]        # current month
    
    # --- Create DataFrame ---
    df = pd.DataFrame({'Month': months, 'Churn Rate (%)': churn_rates})

    rcParams['font.family'] = 'Comic Sans MS'
    rcParams['font.size'] = 7  # Adjust size as needed

    # --- Plot ---
    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Line with circular markers
    ax.plot(df['Month'], df['Churn Rate (%)'], color='skyblue', linewidth=1.5, marker='o', markersize=3)
    
    # Shaded area below the line
    ax.fill_between(df['Month'], df['Churn Rate (%)'], color='skyblue', alpha=0.2)
    
    # Remove chart borders (spines)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    # Styling with Comic Sans
    ax.set_title("Churn Trend Over Time")
    ax.set_ylabel("Churn Rate (%)")
    #ax.set_xlabel("Month")
    ax.grid(True, linestyle='-', alpha=0.5)
    plt.xticks(rotation=0)
    fig.tight_layout()  # ensures consistent padding
    
    # --- Show Plot ---
    st.pyplot(fig)

# display top 3 churn features
def topChurnFeatures(df, total_loss):
    # --- Load or simulate SHAP values ---
    # get model, X    
    with open(MODEL_PATH_T21, "rb") as f:
        model = pickle.load(f) 

    X = df.drop(['Churn'], axis=1)

    # --- Sample background data ---
    background = X.sample(n=100, random_state=42)
    
    #  Define explainer 
    explainer = shap.Explainer(model.predict, background)
    
    #  Cache SHAP computation 
    @st.cache_resource
    def compute_shap_values(X_sample):
        return explainer(X_sample)
    
    #  Compute SHAP values 
    shap_values = compute_shap_values(X[:50])
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    
    # Get top 3 features 
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    top_features = mean_abs_shap.head(3) 

    top_features_percent = (top_features / top_features.sum()) * 100
    #top_features_percent = top_features_percent.round(2)  # optional: round to 2 decimal places
    top_features = top_features_percent
    
    # --- Traffic light colors: Tomato (most impactful), Yellow, Green ---
    colors = ['tomato', 'orange', 'lightgreen']

    # Calculate revenue loss per feature
    revenue_loss_per_feature = (top_features / 100) * total_loss
    revenue_labels = [f"${val:,.0f}" for val in revenue_loss_per_feature]
    
    # --- Bar chart ---
    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)

    # Remove chart borders
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    # Plot horizontal bar chart
    bars = ax.barh(top_features.index, top_features.values, color=colors)
    
    # Add revenue loss labels to each bar
    # Add revenue loss labels inside each bar
    for bar, label in zip(bars, revenue_labels):
        width = bar.get_width()
        ax.text(width * 0.5, bar.get_y() + bar.get_height()/2, label,
                va='center', ha='center', fontsize=10, color='white', fontweight='bold')
    
    # Axis labels and title
    ax.set_xlabel("Churn Impact (%)")
    ax.set_title("Top 3 Churn Features and Revenue Loss")
    
    fig.tight_layout()
    st.pyplot(fig)



#Display churn risk distribution
def run_risk():
    #get the prediction model 
    with open(MODEL_PATH_T21, "rb") as f:
        model = pickle.load(f)   
    
    #load encoded dataset    
    df_encoded = load_dataset.run()  #this function returnes encoded dataset with 22 features 
    df_encoded = df_encoded[df_encoded["Churn"] == 0].copy() # we only interested in those who didn't churn
    df_encoded = df_encoded.drop(['Churn'], axis=1)   

    #load original dataset
    df = pd.read_csv(DATA_PATH)
    df = df[df['Churn'] == 'No']
    df.rename(columns={"tenure": "Months"}, inplace=True)     
    
    # Predict probabilities
    churn_probs = model.predict_proba(df_encoded)[:, 1]       
    
    # Add the prediction back into your DataFrame
    df_encoded["churn_probability"] = churn_probs
    #Add it to the original dataset
    df["churn_probability"] = churn_probs
        
    
    #Set the risk thresholds via streamlit slider for a dynamic input
    #st.sidebar.header("Set Risk Thresholds")
    
    #high_threshold = st.sidebar.slider("High Risk Threshold", min_value=0.4, max_value=0.8, value=0.5, step=0.01)
    #medium_threshold = st.sidebar.slider("Medium Risk Threshold", min_value=0.2, max_value=high_threshold, value=0.3, step=0.01)
    col1, col2 = st.columns(2)

    with col1:
        high_threshold = st.slider(
            "🔴 High Risk Threshold",
            min_value=0.4,
            max_value=0.8,
            value=0.5,
            step=0.01
        )
    
    with col2:
        medium_threshold = st.slider(
            "🟠 Medium Risk Threshold",
            min_value=0.2,
            max_value=high_threshold,
            value=0.3,
            step=0.01
        )

    
    #set risk tires and generat tags
    def categorize_risk(prob):
        if prob >= high_threshold:
            return "High Risk 🚨"
        elif prob >= medium_threshold:
            return "Medium Risk ⚠️"
        else:
            return "Low Risk ✅"
            
    df_encoded["risk_category"] = df_encoded["churn_probability"].apply(categorize_risk)
    #apply it to the original data as well
    df["risk_category"] = df["churn_probability"].apply(categorize_risk)
    
    #visualize in streamlit   
    
    risk_counts = df_encoded["risk_category"].value_counts().reset_index()

    col1, col2 = st.columns(2)
    # --- Churn Distribution ---
    with col1:
        # Define traffic light colors
        color_map = {
            "High Risk 🚨": "red",
            "Medium Risk ⚠️": "orange",
            "Low Risk ✅": "green"
        }
        
        # Create pie chart with custom colors
        st.subheader("Churn Risk Distribution")
        fig = px.pie(
            risk_counts,
            names="risk_category",
            values="count",
            title="",
            color="risk_category",
            color_discrete_map=color_map
        )
        
        st.plotly_chart(fig)  
        #high_threshold = st.slider("High Risk Threshold", min_value=0.4, max_value=0.8, value=0.5, step=0.01)
        #medium_threshold = st.slider("Medium Risk Threshold", min_value=0.2, max_value=high_threshold, value=0.3, step=0.01)
        
    
    # --- Risk tier ---
    with col2:        
        risk_counts = df_encoded["risk_category"].value_counts()        
        st.subheader("Risk Tier Distribution")
        
        for tier in ["High Risk 🚨", "Medium Risk ⚠️", "Low Risk ✅"]:
            count = risk_counts.get(tier, 0)
            percent = count / len(df_encoded)
            st.write(f"{tier}: {count} customers")
            st.progress(percent)
            

            
    
    
    # --- Toggle visibility ---
    show_risk_export = st.toggle("📂 Show Risk Tier Export Panel", value=False)
    
    if show_risk_export:
        st.subheader("View Customers by Risk Tier")
        
        selected_tier = st.selectbox("Choose a risk category", ["High Risk 🚨", "Medium Risk ⚠️", "Low Risk ✅"])
        filtered_df = df[df["risk_category"] == selected_tier]
        
        st.dataframe(filtered_df)
    
        # Export data using a button
        buffer = io.StringIO()
        filtered_df.to_csv(buffer, index=False)
    
        st.download_button(
            label=f"Export {selected_tier} Customers",
            data=buffer.getvalue(),
            file_name=f"{selected_tier.replace(' ', '_').replace('🚨','').replace('⚠️','').replace('✅','').lower()}_customers.csv",
            mime="text/csv"
        )   



#Main App
def run():   

    #Load dataset
    df = load_dataset.run()
    churned_df = df[df['Churn'] == 1] # Churned customers
    total_loss = churned_df['MonthlyCharges'].sum()

    # Format the number for readability
    formatted_loss = f"${total_loss:,.2f}"
    
    # Display as a KPI card
    st.markdown(f"""
        <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="margin-bottom:5px;">💸 Total Revenue Lost from Churned Customers - August 2025</h4>
            <h2 style="color:#d00000; margin-top:0;">{formatted_loss}</h2>
        </div>
    """, unsafe_allow_html=True)




    #showing Churn rate overtime and top 3 churn factors
    col1, col2 = st.columns(2)
    # --- Churn Rate Timeline ---
    with col1:
        st.subheader("Churn Rate Timeline")
        churnRateTimeline(churned_df, df)
    
    # --- Top Churn Features ---
    with col2:
        st.subheader("Top Churn Features")
        topChurnFeatures(df, total_loss)

    run_risk()        

    
    

    

