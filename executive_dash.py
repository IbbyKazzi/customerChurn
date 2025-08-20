import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import openai

#Load dataset module
import load_dataset

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
            st.write(f"Processing cluster {row['cluster']}...")
            segment = llm_cluster_description(row)
            segment_profiles.append(segment)
            time.sleep(20)

    with open(CACHE_PATH, "w") as f:
        json.dump(segment_profiles, f)

    return segment_profiles

#Main App
def run():
    if "prev_n_clusters" not in st.session_state:
        st.session_state["prev_n_clusters"] = None
    st.subheader("📊 Telco Churn Segmentation")

    #Load dataset
    df = load_dataset.run()  

   
    df['Contract_Month-to-month'] = (df['Contract'] == 0).astype(int)
    df['Contract_One_Year'] = (df['Contract'] == 1).astype(int)
    df['InternetService_Fiber optic'] = (df['InternetService'] == 1).astype(int)
    df['TechSupport_No'] = (df['TechSupport'] == 0).astype(int)
    df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 2).astype(int)

    st.write(df)


    #Feature & Cluster Selection
    available_features = [
        'Months', 'MonthlyCharges', 'TotalCharges',
        'Contract_Month-to-month', 'Contract_One_Year',
    'InternetService_Fiber optic', 'TechSupport_No',
    'PaymentMethod_Electronic check'
    ]

    selected_features = available_features
    #st.multiselect(
    #    "Select features for clustering",
    #    options=available_features,
    #    default=['Months', 'MonthlyCharges', 'TotalCharges']
    #)

    n_clusters = st.slider("Select number of clusters", 2, 10, 5)
    # Detect cluster count change
    force_refresh = False
    if st.session_state["prev_n_clusters"] is not None and st.session_state["prev_n_clusters"] != n_clusters:
        force_refresh = True
        st.warning("Cluster count changed — segment profiles will be regenerated.")
    st.session_state["prev_n_clusters"] = n_clusters

    if force_refresh:
        if len(selected_features) < 2:
            st.warning("Please select at least two features.")
            st.stop()

        X_cluster = df[selected_features]
        X_scaled = StandardScaler().fit_transform(X_cluster)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        cluster_summary = df.groupby('cluster').agg({
            'Churn': 'mean',
            'Months': 'mean',
            'MonthlyCharges': 'mean',
            'TotalCharges': 'mean',
            'Contract_Month-to-month':'mean',
            'Contract_One_Year':'mean',
            'TechSupport_No':'mean',
            'PaymentMethod_Electronic check':'mean',
            'InternetService_Fiber optic':'mean'
        }).reset_index()

        st.session_state["cluster_summary"] = cluster_summary
        st.session_state["df"] = df
        #st.success("Clustering complete. Proceed to preview.")

    #Preview Clusters
    #if "cluster_summary" in st.session_state:
    #    st.subheader("📈 Cluster Distribution")
    #    cluster_counts = st.session_state["df"]['cluster'].value_counts().sort_index()
    #    st.bar_chart(cluster_counts)

    import plotly.express as px

    st.subheader("🟢 Cluster Preview (Dot Chart)")
    
    #df = st.session_state.get("df")
    if df is not None and "cluster" in df.columns:
    
        # Feature selectors
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        x_feature = st.selectbox("Select X-axis feature", numeric_cols, index=numeric_cols.index("MonthlyCharges") if "MonthlyCharges" in numeric_cols else 0)
        y_feature = st.selectbox("Select Y-axis feature", numeric_cols, index=numeric_cols.index("Months") if "Months" in numeric_cols else 1)
    
        # Optional filter by cluster
        unique_clusters = sorted(df["cluster"].unique())
        selected_clusters = st.multiselect("Filter clusters", unique_clusters, default=unique_clusters)
    
        filtered_df = df[df["cluster"].isin(selected_clusters)]
    
        # Plot
        fig = px.scatter(
            filtered_df,
            x=x_feature,
            y=y_feature,
            color=filtered_df["cluster"].astype(str),
            title=f"Clusters by {x_feature} vs {y_feature}",
            labels={"cluster": "Cluster"},
            hover_data=["cluster", x_feature, y_feature]
        )
        st.plotly_chart(fig, use_container_width=True)

        
        st.subheader("📊 Cluster Summary")
        st.dataframe(st.session_state["cluster_summary"])    
        
    #st.write(force_refresh)
    
    if "cluster_summary" in st.session_state and st.button("🧠 Generate GPT Segment Descriptions"):        
        segment_profiles = generate_segment_profiles(
            st.session_state["cluster_summary"],
            force_refresh=True
        )
        
        # Validate length before assignment
        if len(segment_profiles) != len(st.session_state["cluster_summary"]):
            st.error(f"Expected {len(st.session_state['cluster_summary'])} segment profiles, but got {len(segment_profiles)}.")
            st.stop()
        
        st.session_state["cluster_summary"]["Segment_Profile"] = segment_profiles
        st.success("Segment profiles ready.")

        # Display segment cards
        for idx, row in st.session_state["cluster_summary"].iterrows():
            st.markdown(f"### 🧠 Cluster {row['cluster']}: {row['Segment_Profile'].split(':')[0]}")
            st.markdown(f"**📝 Description:** {row['Segment_Profile'].split(':')[1].strip()}")

            col1, col2, col3 = st.columns(3)
            col1.metric("📉 Churn Rate", f"{row['Churn']:.2%}")
            col2.metric("📆 Avg Tenure", f"{row['Months']:.1f} months")
            col3.metric("💰 Monthly Charges", f"${row['MonthlyCharges']:.2f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("🌐 Fiber Usage", f"{row['InternetService_Fiber optic']*100:.1f}%")
            col5.metric("🛠️ No Tech Support", f"{row['TechSupport_No']*100:.1f}%")
            col6.metric("💳 Electronic Check", f"{row['PaymentMethod_Electronic check']*100:.1f}%")

            st.markdown("---")

        st.download_button("📥 Download Summary", st.session_state["cluster_summary"].to_csv(index=False), "cluster_summary.csv")

#Run the app
#run()
