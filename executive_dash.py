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



#Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

#Cache path
CACHE_PATH = "segment_profiles.json"

#GPT Segment Description Generator
def llm_cluster_description(row):
    prompt = (
        "You are an expert in telecommunications customer analytics. "
        "Based on the following cluster characteristics from a telco churn dataset, "
        "assign one short business-friendly segment name (2–4 words) and a brief description.\n"
        f"- Average churn rate: {row['Churn']:.2f}\n"
        f"- Average tenure: {row['Months']:.1f} months\n"
        f"- Average monthly charges: {row['MonthlyCharges']:.2f}\n"
        f"- % Month-to-month contract: {row['Contract_Month-to-month']*100:.1f}%\n"
        f"- % Contract_One_Year contract: {row['Contract_One_Year']*100:.1f}%\n"
        f"- % using Fiber: {row['InternetService_Fiber optic']*100:.1f}%\n"
        f"- % without Tech Support: {row['TechSupport_No']*100:.1f}%\n"
        f"- % using Electronic check: {row['PaymentMethod_Electronic check']*100:.1f}%\n\n"
        "Use one of the following label styles if it applies: "
        "'New Cost-Sensitive Users', 'Tech-Avoidant Casuals', 'Bundled Value Seekers', "
        "'Contract Expiry Risks', 'Electronic-Check Churners', 'Loyal Seniors', 'Streaming-Heavy Customers'. "
        "If none of these apply, generate a similar style. "
        "Return output in the format: <Segment Name>: <Description>"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:        
        st.error(f"OpenAI error: {str(e)}")
        return f"Error: {str(e)}"

#GPT Segment Generation with Caching
def generate_segment_profiles(df_summary, force_refresh):
    if not force_refresh and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)

    segment_profiles = []
    with st.spinner("Generating segment profiles..."):
        for idx, row in df_summary.iterrows():
            #st.write(f"Processing cluster {row['cluster']}...")
            segment = llm_cluster_description(row)
            segment_profiles.append(segment)
            time.sleep(0.1)

    with open(CACHE_PATH, "w") as f:
        json.dump(segment_profiles, f)

    return segment_profiles

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
def topChurnFeatures(df):
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
    
    # --- Bar chart ---
    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)
    # Remove chart borders (spines)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
        
    top_features.plot(kind='barh', color=colors, ax=ax)
    ax.set_xlabel("Churn Impact (%)")
    ax.set_title("Top 3 Churn Features")
    fig.tight_layout()  # ensures consistent padding
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
    churned_df = df[df['Churn'] == 1]

    #showing Churn rate overtime and top 3 churn factors
    col1, col2 = st.columns(2)
    # --- Churn Rate Timeline ---
    with col1:
        st.subheader("Churn Rate Timeline")
        churnRateTimeline(churned_df, df)
    
    # --- Top Churn Features ---
    with col2:
        st.subheader("Top Churn Features")
        topChurnFeatures(df)

    run_risk()        

    # Total revenue lost from churned customers    
    total_loss = churned_df['MonthlyCharges'].sum()
    

    # Sidebar controls
    st.sidebar.header("Churn Impact Simulator")
    
    retention_slider = st.sidebar.slider("Assumed retention rate (%)", 0, 100, 50)
    
    retained_fraction = retention_slider / 100
    adjusted_loss = total_loss * (1 - retained_fraction)
    
    st.sidebar.metric("💸 Revenue Loss", f"${adjusted_loss:,.2f}")
    
    #st.sidebar.markdown("<div style='height: 500px;'></div>", unsafe_allow_html=True)

