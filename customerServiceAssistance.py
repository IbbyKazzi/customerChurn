#This assistance is designed to narrate some insights by the model to help customer service team decision making
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import time
from settings import MODEL_PATH_T3, MODEL_PATH_T21, DATA_PATH

def run(customer, shap_values, X, contract_map, df):  
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
     
    # Compute mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Create a DataFrame for easy sorting
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(local_shap)
    }).sort_values(by='importance', ascending=False)

    
    # Display top N features
    top_n = 5
    top_features = feature_importance.head(top_n)
    customer["recommended_plan"] = recommended   
    customer["top_features"] = top_features.feature   
    st.write(top_features)
    # Inject custom CSS to position the chat box
    st.markdown("""
        <style>
        .chat-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 50px;
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
        message = (
                "👋 **Hi, I'm ChurnMate!** I'm here to help you understand churn risks and recommend retention strategies.\n\n"
                "🧠 **ChurnMate:** Here's what I found:\n\n"
                + summarize_customer(customer)
            )
        if "show_message" not in st.session_state:
            st.session_state["show_message"] = True            
            showIntro(message, 0.005)
        else:
           showIntro(message, 0)        

        if customer["churn_probability"] > 0.5:
              st.warning("⚠️ **ChurnMate Alert**: This customer is at very high risk. Consider immediate outreach.")        
        
        question = st.text_input("Ask me anything about this customer or churn trends:")
        if question:
          response = generate_response(question, customer, shap_values, contract_map, df)            
          

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
    
    return  (
        f"👋 Hey there! I’ve analyzed **{customer_name}**.\n\n"
        f"🔍 **Churn Risk**: {risk_level.capitalize()} ({churn_prob:.1%})\n\n"
        #f"📌 **Key Factors**: {factors}\n\n"
        #f"💡 **Suggestion**: Consider offering the **{plan_suggestion}** plan to improve retention."
    )
    #showResponse(response)    

def generate_response(question, data, shap_values, contract_map, df):
    question = question.lower()
    churn_prob = data.get("churn_probability", 0.0)
    top_features = data.get("top_features", [])
    #plan = data.get("recommended_plan", "Premium")
    plan = data["recommended_plan"]
    monthlyCharges = data["monthlyCharges"]
  
    if "why" in question:
        reasons = ", ".join(top_features[:2])
        response =  ( 
            f"🧠 **ChurnMate:** "
            f"This customer is likely to churn due to {reasons}. "
            f"Their churn probability is {churn_prob:.1%}, which is considered {'high' if churn_prob > 0.5 else 'moderate' if churn_prob > 0.25 else 'low'}."
        )
        showResponse(response)

    elif "recommend" in question or "suggest" in question:
        response =  (
            f"🧠 **ChurnMate:** "
            f"I recommend offering the **{plan}** plan. "
            f"It typically reduces churn by offering better value and longer contract terms.\n"
            f"You can try the below plans and contracts combination to determine churn risk for this customer."
        )
        showResponse(response)
        showRecommandation(contract_map, data["tenure"])

    elif "risk" in question or "chance" in question:
        response =  (            
            f"The churn risk for this customer is **{churn_prob:.1%}**. "
            f"This is based on factors like {', '.join(top_features[:2])}."
        )
        showResponse(response)

    elif "features" in question or "factors" in question:       
        response =   (
          f"🧠 **ChurnMate:** "
          f"The top factors influencing churn are: {', '.join(top_features)}. "
          f"These features have the highest SHAP impact on the prediction. Click on the toggle below to view more details."
        )
        showResponse(response)         
        # Show waterfall plot if toggle is activated
        if st.toggle("Show churn factor waterfall"):
            st.session_state["show_response"] = False
            st.markdown("### Factors of Churn")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values, show=False)
            st.pyplot(fig)

    elif "details" in question or "show customer details" in question:
      response =   (
          f"🧠 **ChurnMate:** "
          f"Below a full list of the customer details."         
      )
      showResponse(response)
      # Show waterfall plot if toggle is activated
      i = data["index"]
      st.write(df.iloc[i])

    elif "plan" in question:
        response = (  
            f"🧠 **ChurnMate:** "
            f"The current plan is **{data.get('current_plan', 'Unknown')}**, "
            f"but switching to **{plan}** may reduce churn risk."
        )
        showResponse(response)
            
    elif "price" in question or "paying" in question or "charges" in question:      
       return (           
           f"The current monthly charges are **${monthlyCharges}**, "
           f"but switching to **{plan}** may reduce churn risk."
       )
    elif "strategy" in question or "retension" in question:       
        strategy = generate_strategy(data["churn_probability"])
        showResponse(strategy)
           

    else:
        return (  
            f"🧠 **ChurnMate:** "
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
        return (
          f"🧠 **ChurnMate:** "
          "Offer a long-term discount or loyalty plan"
        )
    elif churn_risk > 0.25:
        return (          
          f"🧠**ChurnMate: Provide personalized support and flexible options**\n\n"
          f"\u00A0📌 **Why?**\n\n This customer shows signs of hesitation—possibly due to pricing concerns or unmet needs. "
          f"Offering tailored assistance (e.g., live chat, dedicated rep) and adaptable plans (like pausing service, switching tiers, or bundling features) "
          f"can reinforce trust and reduce churn risk. Highlight benefits that align with their usage patterns or preferences.\n\n"
          f"\u00A0🔍 Optional Add-ons:\n\n"
          f"**Usage-based triggers:** Customer has contacted support 3× in the past month—indicates need for proactive outreach.\n\n"
          f"**Plan flexibility:** Offer downgrade or pause options to retain value without forcing cancellation.\n\n"
          f"**Personalization angle:** Reference past interactions or preferences to make support feel relevant and empathetic."
        )
    else:
        return (
          f"🧠 **Maintain current engagement strategy**\n\n"          
          f"\u00A0📌 **Why?**\n\n This customer shows stable usage patterns and low churn risk. "
          f"Current touchpoints—such as monthly check-ins and personalized offers—are effectively sustaining engagement. "
          f"No immediate changes are needed, but continue monitoring for shifts in behavior."
        )

def showIntro(message, delay):
    if delay == 0:
        st.info(message)
    else:
        # Create a placeholder for the success box
        placeholder = st.empty()      
        typed_text = ""
        for char in message:
            typed_text += char
            # Render the full success box once, updating its content
            placeholder.info(typed_text)
            time.sleep(delay)
def showResponse(response):    
    message = response      
    # Create a placeholder for the success box
    placeholder = st.empty()      
    typed_text = ""
    for char in message:
        typed_text += char
        # Render the full success box once, updating its content
        placeholder.success(typed_text)
        time.sleep(0.005)
            
def showRecommandation(contract_map, tenure):
  #Recommend Plan  
  # dropdown to allow manual override
  available_plans = ["Basic", "Standard", "Premium", "Family", "Enterprise"]
  override = st.selectbox("🧾 Override Plan Suggestion", options=available_plans)
  # get the new prob of this customer for the selected plan
  #new_prob = plan_churn_df.loc[plan_churn_df["Plan"] == override, "Churn Probability"].values[0]
    
  # add radio widget
  selected_contract = st.radio(
    "📝 Select Contract",
    ["Month-to-month", "One year", "Two year"],
    horizontal=True
  )

  customer_contract = contract_map[selected_contract]
  new_prob = get_newProb(override, tenure, customer_contract)    
  st.markdown(f"**Estimated Churn Probability for {override} Plan:** {new_prob:.2%}")     
     
  #check recommandation outcome
  st.markdown("#### Recommendation Outcome")
  st.radio("Was the recommendation accepted?", ["Yes", "No", "Pending"])
  st.text_area("Agent Notes")
  st.button("Submit Feedback")

#get new prob of the overrided plan
def get_newProb(val, tenure, contract):
    with open(MODEL_PATH_T3, "rb") as f:
        model = pickle.load(f)    
    monthly_charges = [25, 55, 85, 115, 145]    
    plan_labels = ["Basic", "Standard", "Premium", "Family", "Enterprise"]
    plan_charge_df = pd.DataFrame({
    "Plan": plan_labels,
    "Monthly Charge": monthly_charges
    })
    selected_charge = plan_charge_df.loc[plan_charge_df["Plan"] == val, "Monthly Charge"].values[0]   

    input_data = np.array([[tenure, selected_charge, contract]])
    prediction = model.predict_proba(input_data)
    new_churn_probability = prediction[0][1]
    return new_churn_probability


