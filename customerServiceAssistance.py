#This assistance is designed to narrate some insights by the model to help customer service team decision making
import streamlit as st
import pandas as pd
import numpy as np

def run():
  ############Chat box assistance Plan recommandation################
    plans = {
        "Basic": 25,
        "Standard": 55,
        "Premium": 85,
        "Family": 115,
        "Enterprise": 145
    }
    churn_risk = {
        "Basic": 0.25,
        "Standard": 0.18,
        "Premium": 0.12,
        "Family": 0.08,
        "Enterprise": 0.05
    }
    recommended = None
    best_score = float("inf")    
    for plan, price in plans.items():
        churn = churn_risk[plan]
        price_diff = abs(price - monthly_charges)
        score = churn * 100 + price_diff  # weighted score
    
        if score < best_score:
            best_score = score
            recommended = plan
    
    st.write(f"✅ Recommended Plan: {recommended}")
    
    # Sidebar header
    st.sidebar.header("📦 Available Plans")
    
    # Display each plan
    for plan, price in plan_prices.items():
        st.sidebar.write(f"**{plan}**: {price}")

    #Assistant churnMate ########################################
    # Compute mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Create a DataFrame for easy sorting
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)
    
    # Display top N features
    top_n = 5
    top_features = feature_importance.head(top_n)

    customer = {
        "name": selected_customer_id,
        "churn_probability": churn_probability,
        "top_feature": top_features["feature"],
        "recommended_plan": recommended 
    }

    # Inject custom CSS to position the chat box
    st.markdown("""
        <style>
        .chat-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            max-height: 500px;
            background-color: #f9f9f9;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            padding: 15px;
            overflow-y: auto;
            z-index: 9999;
        }
        .chat-header {
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create the chat box container

    with st.container():
        st.markdown("👋 **Hi, I'm ChurnMate!** I'm here to help you understand churn risks and recommend retention strategies.")
        #uploaded_file = st.file_uploader("Upload customer data")
        #selected_segment = st.selectbox("Choose a customer segment", ["All", "High Risk", "Premium Plan"])
        st.markdown("🧠 **ChurnMate:** Here's what I found:")
        st.markdown(summarize_customer(customer))        
        
        question = st.text_input("Ask me anything about this customer or churn trends:")
        if question:
            response = generate_response(question, customer)
            st.markdown(f"💬 **ChurnMate:** {response}")
    
        if customer["churn_probability"] > 0.7:
            st.warning("⚠️ ChurnMate Alert: This customer is at very high risk. Consider immediate outreach.")
    
        if st.button("Generate Retention Strategy"):
            strategy = generate_strategy(customer["churn_probability"])
            st.success(f"💡 ChurnMate Suggests: {strategy}")

        ############Chat box########################       
