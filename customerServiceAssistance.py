#This assistance is designed to narrate some insights by the model to help customer service team decision making
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run(customer, shap_values, X,i):
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
        price_diff = abs(price - customer["monthlyCharges"])
        score = churn * 100 + price_diff  # weighted score
    
        if score < best_score:
            best_score = score
            recommended = plan
    
    #st.write(f"✅ Recommended Plan: {recommended}")
    
    # Sidebar header
    st.sidebar.header("📦 Available Plans")
    
    

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
    customer["recommended_plan"] = recommended   
    customer["top_features"] = top_features.feature
    #st.write(top_features)

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

def summarize_customer(customer):
    churn_prob = customer["churn_probability"]
    top_factors = customer["top_features"]
    plan = customer["recommended_plan"]
    
    return assistant_response(customer["name"], churn_prob, top_factors, plan)

def assistant_response(customer_name, churn_prob, top_features, plan_suggestion):
    risk_level = (
        "high" if churn_prob > 0.5 else
        "moderate" if churn_prob > 0.25 else
        "low"
    )
    factors = ", ".join(top_features[:5])
    
    return (
        f"👋 Hey there! I’ve analyzed **{customer_name}**.\n\n"
        f"🔍 **Churn Risk**: {risk_level.capitalize()} ({churn_prob:.1%})\n\n"
        #f"📌 **Key Factors**: {factors}\n\n"
        #f"💡 **Suggestion**: Consider offering the **{plan_suggestion}** plan to improve retention."
    )

def generate_response(question, data):
    question = question.lower()
    churn_prob = data.get("churn_probability", 0.0)
    top_features = data.get("top_features", [])
    plan = data.get("recommended_plan", "Premium")

    if "why" in question:
        reasons = ", ".join(top_features[:2])
        return (
            f"This customer is likely to churn due to {reasons}. "
            f"Their churn probability is {churn_prob:.1%}, which is considered {'high' if churn_prob > 0.5 else 'moderate' if churn_prob > 0.25 else 'low'}."
        )

    elif "recommend" in question or "suggest" in question:
        return (
            f"I recommend offering the **{plan}** plan. "
            f"It typically reduces churn by offering better value and longer contract terms."
        )

    elif "risk" in question or "chance" in question:
        return (
            f"The churn risk for this customer is **{churn_prob:.1%}**. "
            f"This is based on factors like {', '.join(top_features[:2])}."
        )

    elif "features" in question or "factors" in question:      
      response = (
          f"🧠 **ChurnMate:** "
          f"The top factors influencing churn are: {', '.join(top_features)}. "
          f"These features have the highest SHAP impact on the prediction. Click on the toggle below to view more details."
      )
      st.markdown(response)
      
      # Add spacing
      st.markdown("<div style='margin-top: 20px'></div>", unsafe_allow_html=True)
      
      # Show waterfall plot if toggle is activated
      if st.toggle("Show churn factor waterfall"):
          st.markdown("### Factors of Churn")
          fig, ax = plt.subplots()
          shap.plots.waterfall(shap_values[i], show=False)
          st.pyplot(fig)

    elif "plan" in question:
        return (
            f"The current plan is **{data.get('current_plan', 'Unknown')}**, "
            f"but switching to **{plan}** may reduce churn risk."
        )

    else:
        return (
            "I'm here to help with churn insights! Try asking:\n"
            "- Why is this customer likely to churn?\n"
            "- What plan do you recommend?\n"
            "- What is their churn risk?\n"
            "- What features are driving churn?"
        )

def segment_summary(segment_data):
    avg_churn = segment_data["churn_probability"].mean()
    common_factors = segment_data["top_features"].explode().value_counts().head(2).index.tolist()
    
    return (
        f"📊 In this segment, average churn risk is **{avg_churn:.1%}**.\n"
        f"🔍 Common churn drivers: {', '.join(common_factors)}."
    )

def generate_strategy(churn_risk):
    if churn_risk > 0.5:
        return "Offer a long-term discount or loyalty plan"
    elif churn_risk > 0.25:
        return "Provide personalized support and flexible options"
    else:
        return "Maintain current engagement strategy"



