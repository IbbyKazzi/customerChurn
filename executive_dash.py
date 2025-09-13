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
    fig, ax = plt.subplots()
    
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
    
    # --- Show Plot ---
    st.pyplot(fig)

# display top 3 churn features
def topChurnFeatures(df, X):
    # --- Load or simulate SHAP values ---
    # get model, X    
    with open(MODEL_PATH_T21, "rb") as f:
        model = pickle.load(f) 

    X = df_encoded.drop(['Churn'], axis=1)   
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # --- Compute mean absolute SHAP values ---
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    
    # --- Get top 3 features ---
    top_features = mean_abs_shap.head(3)
    
    # --- Display ---
    st.subheader("📌 Top 3 Features Driving Churn")
    for i, (feature, value) in enumerate(top_features.items(), 1):
        st.markdown(f"**{i}. {feature}** — SHAP impact: `{value:.4f}`")
    
    # --- Optional: Bar chart ---
    fig, ax = plt.subplots()
    top_features.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 3 Churn Features")
    st.pyplot(fig)

    

#Main App
def run():
    if "prev_n_clusters" not in st.session_state:
        st.session_state["prev_n_clusters"] = None
        st.header("📊 Segmentation Strategy Panel")
        st.subheader("Select the granularity of customer clusters required to tailor retention insights:")

    #Load dataset
    df = load_dataset.run()  
    churned_df = df[df['Churn'] == 1]

    #showing Churn rate overtime
    churnRateTimeline(churned_df, df)
    topChurnFeatures(df)
       
    df['Contract_Month-to-month'] = (df['Contract'] == 0).astype(int)
    df['Contract_One_Year'] = (df['Contract'] == 1).astype(int)
    df['InternetService_Fiber optic'] = (df['InternetService'] == 1).astype(int)
    df['TechSupport_No'] = (df['TechSupport'] == 0).astype(int)
    df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 2).astype(int)
    
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
    with st.expander("📘 Why Cluster Granularity Matters"):
         st.markdown("""
         - **Fewer Clusters (2–4)**: Broad segmentation ideal for high-level strategy and executive summaries.
         - **Moderate Clusters (5–7)**: Balanced granularity for tactical planning across departments.
         - **More Clusters (8–10)**: Fine-grained segmentation for targeted interventions and personalized retention tactics.
        
         Adjust the slider to explore how different segmentation levels impact churn insights and tactical recommendations.
        """)

    st.markdown("**Select number of customer clusters required:**")
    n_clusters = st.slider("", 2, 10, 5)
    st.markdown(f"🔎 You’ve selected **{n_clusters}** customer clusters.")

    
    # Detect cluster count change
    if "force_refresh" not in st.session_state:
        st.session_state["force_refresh"] = False
    
    if st.session_state["prev_n_clusters"] is not None and st.session_state["prev_n_clusters"] != n_clusters:
        st.session_state["force_refresh"] = True
    st.session_state["prev_n_clusters"] = n_clusters
    

    if st.session_state["force_refresh"]:
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
    if "cluster_summary" in st.session_state:
        show_charts = st.toggle("Show Cluster Charts and Summary", value=False)
    
        if show_charts:
            st.subheader("📈 Cluster Distribution")
            cluster_counts = st.session_state["df"]['cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)
    
            st.subheader("📊 Cluster Summary")
            df = st.session_state["cluster_summary"].reset_index(drop=True)
            df.index = [''] * len(df)  # Blank out index labels
            st.dataframe(st.session_state["cluster_summary"], hide_index=True)


        
    #st.write(force_refresh)
    #if "cluster_summary" in st.session_state and st.button("🧠 Show GPT Segment Descriptions") or st.session_state["force_refresh"]:
    
    if "cluster_summary" in st.session_state or st.session_state["force_refresh"]:
        segment_profiles = generate_segment_profiles(
            st.session_state["cluster_summary"],
            force_refresh=st.session_state["force_refresh"]
        )
    
        # Reset refresh flag after use
        st.session_state["force_refresh"] = False
    
        if len(segment_profiles) != len(st.session_state["cluster_summary"]):
            st.error(f"Expected {len(st.session_state['cluster_summary'])} segment profiles, but got {len(segment_profiles)}.")
            st.stop()
    
        st.session_state["cluster_summary"]["Segment_Profile"] = segment_profiles
        st.success("GPT-Powered Segment Insights Available")
        
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

    # Total revenue lost from churned customers    
    total_loss = churned_df['MonthlyCharges'].sum()
    

    # Sidebar controls
    st.sidebar.header("Churn Impact Simulator")
    
    retention_slider = st.sidebar.slider("Assumed retention rate (%)", 0, 100, 50)
    
    retained_fraction = retention_slider / 100
    adjusted_loss = total_loss * (1 - retained_fraction)
    
    st.sidebar.metric("💸 Revenue Loss", f"${adjusted_loss:,.2f}")
    
    #st.sidebar.markdown("<div style='height: 500px;'></div>", unsafe_allow_html=True)

