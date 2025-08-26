import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from settings import MODEL_PATH_T3, MODEL_PATH_T21
from datetime import datetime
import pytz
from datetime import timedelta
import openai
from zoneinfo import ZoneInfo

def call_gpt_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response    

def run():
    # Define plans and contract types
    monthly_charges = [25, 55, 85, 115, 145]
    contract_types = {0: "Month-to-Month", 1: "Yearly", 2: "2-Year"}
    plan_labels = ["Basic", "Standard", "Premium", "Family", "Enterprise"]
    
    def assign_plan(charge):
        if charge <= 39:
            return "Basic"
        elif charge <= 69:
            return "Standard"
        elif charge <= 99:
            return "Premium"
        elif charge <= 129:
            return "Family"
        else:
            return "Enterprise"
        
    with open(MODEL_PATH_T3, "rb") as f:
        model = pickle.load(f)
    
    # Generate churn data across contracts
    results = []
    for contract_code, contract_label in contract_types.items():
        df = pd.DataFrame({
            "MonthlyCharges": monthly_charges,
            "Months": [1]*5,
            "Contract": [contract_code]*5
        })
        df["Plan"] = df["MonthlyCharges"].apply(assign_plan)  
        X = df[["Months", "MonthlyCharges", "Contract"]]
    
        # check which one is better present, chrun classification or churn probability
        
        #df["PredictedChurn"] = model.predict(X)
        # Get churn probability (assuming binary classification: [No Churn, Churn])
        df["ChurnProbability"] = model.predict_proba(X)[:, 1]  # Probability of class '1' (churn)
        
        #churn_by_plan = df.groupby("Plan")["PredictedChurn"].mean().reset_index()
        #churn_by_plan.columns = ["Plan", "Predicted Churn Rate"]
    
        churn_by_plan = df.groupby("Plan")["ChurnProbability"].mean().reset_index()
        churn_by_plan.columns = ["Plan", "Churn Probability"]
        
        churn_by_plan["ContractType"] = contract_label
        results.append(churn_by_plan)
    
    # Combine into matrix
    matrix_df = pd.concat(results)
    matrix_df["Plan"] = pd.Categorical(matrix_df["Plan"], categories=plan_labels, ordered=True)
    matrix_df = matrix_df.sort_values("Plan")
    
    pivot_df = matrix_df.pivot(index="Plan", columns="ContractType", values="Churn Probability").reset_index()
    
    #pivot_df = matrix_df.pivot(index="Plan", columns="ContractType", values="Predicted Churn Rate").reset_index()
    
    # Convert churn rate columns to float
    for col in pivot_df.columns[1:]:  # Skip 'Plan' column
        pivot_df[col] = pd.to_numeric(pivot_df[col], errors="coerce")
    
    #st.write(pivot_df.columns)
    
    #streamlit UI with plan selector
    
    st.subheader("Churn Rate by Plan and Contract Type")
    
    # Dropdown to select plan
    selected_plan = st.selectbox("Select a Monthly Plan", plan_labels)
    
    # Filter data for selected plan
    plan_data = matrix_df[matrix_df["Plan"] == selected_plan]
    
    # Bar chart across contract types
    #fig = px.bar(
    #    plan_data,
    #    x="ContractType",
    #    y="Churn Probability",
    #    color="Churn Probability",
    #    color_continuous_scale="OrRd",
    #    title=f"Churn Rate for {selected_plan} Plan",
    #    labels={"ContractType": "Contract Type", "Predicted Churn Probability": "Churn Probability"}
    #)
    #st.plotly_chart(fig)
    
    # Convert churn probability to percentage
    plan_data["Churn Probability (%)"] = plan_data["Churn Probability"] * 100
    
    # Create bar chart using the new percentage column
    fig = px.bar(
        plan_data,
        x="ContractType",
        y="Churn Probability (%)",
        color="Churn Probability (%)",
        color_continuous_scale="OrRd",
        title=f"Churn Rate for {selected_plan} Plan",
        #labels={
        #    "ContractType": "Contract Type",
        #    "Churn Probability (%)": "Churn Probability (%)"
        #},
        text=plan_data["Churn Probability (%)"].apply(lambda x: f"{x:.2f}%")
    )
    
    fig.update_traces(textposition="inside")
    st.plotly_chart(fig, use_container_width=True)
    
    
    #Add heatmap with matrix view
    st.subheader("📋 Full Churn Rate Matrix")
    #st.dataframe(pivot_df.style.format("{:.2%}"), hide_index=True)
    #st.dataframe(pivot_df, hide_index=True)
    
    #st.write(pivot_df.dtypes)
    #st.write(pivot_df.head())
    
    # Build formatter dictionary for non-'Plan' columns
    formatters = {
        col: "{:.2%}" for col in pivot_df.columns if col != "Plan"
    }
    
    # Apply style with targeted formatting
    styled_df = pivot_df.style.format(formatters).background_gradient(cmap="Reds", axis=None)
    # Display in Streamlit
    st.dataframe(styled_df, hide_index=True)  

    #connect to openAi and get insight of the full rate matrix
    # Convert pivot_df to markdown-style table
    table_str = pivot_df.to_markdown(index=False)
    
    # Create a prompt
    prompt = f"""
    We are analyzing churn risk across different subscription plans and contract types. Plans range from Basic to Premium, and contracts include Monthly, One-Year, and Two-Year options.
    
    Here is the churn probability matrix:
    
    {table_str}
    
    Please provide insights, highlight any concerning patterns, and suggest actions to reduce churn.
    """
    
    #Load OpenAI API key
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    if "gpt_response" not in st.session_state:
        st.session_state.gpt_response = None

    if "gpt_timestamp" not in st.session_state:
        st.session_state.gpt_timestamp = None

    refresh = st.button("🔄 Refresh GPT Response")
    
    if st.session_state.gpt_response is None or refresh:
        with st.spinner("Calling GPT..."):
            response = call_gpt_api(prompt)
            st.session_state.gpt_response = response
            sydney_tz = pytz.timezone("Australia/Sydney")
            st.session_state.gpt_timestamp = datetime.now(sydney_tz)
    
    # Show GPT insight
    #st.subheader("🧠 GPT Analysis")
    insight_text = st.session_state.gpt_response
    with st.expander("🧠 Click to view GPT-generated insights"):
        st.markdown(insight_text)   
        now = datetime.now(pytz.utc)
        sydney_tz = pytz.timezone("Australia/Sydney")
        now_sydney = now_utc.astimezone(sydney_tz)
        age = now_sydney - st.session_state.gpt_timestamp.replace(tzinfo=pytz.utc)
        
        if age < timedelta(minutes=5):
            freshness = "🟢 Fresh"
        elif age < timedelta(hours=1):
            freshness = "🟡 Stale"
        else:
            freshness = "🔴 Outdated"
        
        st.markdown(f"{freshness} · Last updated: {st.session_state.gpt_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add download    
    st.download_button("Download Insight", insight_text, file_name="churn_insight.txt")

    


    
    #Dispaly a heat map for all plans
    st.subheader("🧯 Heatmap View")
    #below will display heatmap values as decimals
    #heatmap_df = pivot_df.set_index("Plan")
    #fig = px.imshow(
    #    heatmap_df,
    #    labels=dict(x="Contract Type", y="Plan", color="Churn Rate"),
    #    color_continuous_scale="Reds",
    #    text_auto=True,
    #    width=900,
    #    height=600
    #)
    #st.plotly_chart(fig)
    
    #Below will dispaly heatmap as percentages
    
    # Convert values to percentage
    percent_values = pivot_df.drop("Plan", axis=1).values * 100
    
    # Create annotations for each cell
    annotations = [
        dict(
            text=f"{percent_values[i][j]:.1f}%",
            x=pivot_df.columns[1:][j],
            y=pivot_df["Plan"][i],
            xref="x1",
            yref="y1",
            showarrow=False,
            font=dict(color="black", size=12)
        )
        for i in range(percent_values.shape[0])
        for j in range(percent_values.shape[1])
    ]
    
    # Plot heatmap
    fig = go.Figure(data=go.Heatmap(
        z=percent_values,
        x=pivot_df.columns[1:],
        y=pivot_df["Plan"],
        colorscale="Reds",
        showscale=True,
        hoverinfo="text",
        text=[[f"{val:.2f}%" for val in row] for row in percent_values]
    ))
    
    fig.update_layout(
        title="Churn Risk by Plan",
        xaxis_title="Contract Type",
        yaxis_title="Monthly Plan",
        annotations=annotations
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display plan values in sidebar
    plan_prices = {
        "Basic": "$25",
        "Standard": "$55",
        "Premium": "$85",
        "Family": "$115",
        "Enterprise": "$145"    
    }
    st.sidebar.header("📦 Available Plans")
    for plan in pivot_df["Plan"].unique():
        price = plan_prices.get(plan, "Price not available")
        st.sidebar.write(f"- {plan}: {price}")
