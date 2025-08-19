import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import openai


def run():
    
    #Load OpenAI API key
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    
    #Load dataset
    import load_dataset  # Your custom module
    df = load_dataset.run()
    
    # Feature selection
    available_features = [
        'Months', 'MonthlyCharges', 'TotalCharges',
        'Contract', 'InternetService', 'TechSupport',
        'PaymentMethod'
    ]
    
    selected_features = st.multiselect(
        "Select features for clustering",
        options=available_features,
        default=['Months', 'MonthlyCharges', 'TotalCharges']
    )
    
    #Cluster count slider
    n_clusters = st.slider("Select number of clusters", 2, 10, 5)
    
    # Validate input
    if len(selected_features) < 2:
        st.warning("Please select at least two features for clustering.")
        st.stop()
    
    # Regenerate Clusters
    if st.button("🔄 Regenerate Clusters"):
        X_cluster = df[selected_features]
        X_scaled = StandardScaler().fit_transform(X_cluster)
    
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
    
        #Cluster summary
        cluster_summary = df.groupby('cluster').agg({
            'Churn': 'mean',
            'Months': 'mean',
            'MonthlyCharges': 'mean',
            'TotalCharges': 'mean',
            'Contract': 'mean',
            'InternetService': 'mean',
            'TechSupport': 'mean',
            'PaymentMethod': 'mean'
        }).reset_index()
    
        st.subheader("📈 Cluster Distribution")
        cluster_counts = df['cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)
    
        st.subheader("📊 Cluster Summary Preview")
        st.dataframe(cluster_summary)
    
        #GPT Labeling Trigger
        if st.button("🧠 Generate GPT Segment Descriptions"):
            def llm_cluster_description(row):
                prompt = (
                    "You are an expert in telecommunications customer analytics. "
                    "Based on the following cluster characteristics from a telco churn dataset, "
                    "assign one short business-friendly segment name (2–4 words) and a brief description. "
                    "Here are the characteristics:\n"
                    f"- Average churn rate: {row['Churn']:.2f}\n"
                    f"- Average tenure: {row['Months']:.1f} months\n"
                    f"- Average monthly charges: {row['MonthlyCharges']:.2f}\n"
                    f"- % using Fiber: {row['InternetService']*100:.1f}%\n"
                    f"- % without Tech Support: {row['TechSupport']*100:.1f}%\n"
                    f"- % using Electronic check: {row['PaymentMethod']*100:.1f}%\n\n"
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
                    return f"Error: {str(e)}"
    
            segment_profiles = []
            with st.spinner("Generating segment profiles..."):
                for idx, row in cluster_summary.iterrows():
                    st.write(f"Processing cluster {row['cluster']}...")
                    segment = llm_cluster_description(row)
                    segment_profiles.append(segment)
                    time.sleep(20)  # Respect rate limit
    
            cluster_summary['Segment_Profile'] = segment_profiles
    
            #Save to cache
            with open("segment_profiles.json", "w") as f:
                json.dump(segment_profiles, f)
    
            st.success("Segment profiles generated.")
            st.dataframe(cluster_summary)
