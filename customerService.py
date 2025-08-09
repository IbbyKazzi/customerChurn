import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from settings import MODEL_PATH_T3, MODEL_PATH_T21, DATA_PATH

def run():
    #get the prediction model    
    with open(MODEL_PATH_T21, "rb") as f:
        model = pickle.load(f)
    
    #load the dataset
    import load_dataset
    df_encoded = load_dataset.run()  #this function returnes encoded dataset with 22 features  
    X = df_encoded.drop(['Churn'], axis=1)   
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # Load your dataset to extract customer ids
    df = pd.read_csv(DATA_PATH)
    df_filtered = df[df['Churn'] == 'No']
    
    # Extract unique customer IDs
    customer_ids_df = df_filtered['customerID'].reset_index()
    
    
    #set the page menu  Customer-Churn-dataset.csv
    st.sidebar.header("Customer Filter")
    # Choose the customer index
    selected_customer_id = st.sidebar.selectbox("Enter Customer ID", options=customer_ids_df['customerID'])
    
    # Now get the original index from the df
    i = customer_ids_df[customer_ids_df['customerID'] == selected_customer_id]['index'].values[0]   
    
    #get selected customer's tenure,monthly charge and contract and use our prediction model to check churn possibility
    tenure = df.iloc[i]["tenure"]
    monthly_charges = df.iloc[i]["MonthlyCharges"]
    contract = df.iloc[i]["Contract"]
    #get the top 3 prediction model
    with open(MODEL_PATH_T3, "rb") as f:
        model_t3 = pickle.load(f)
    # encode categorical input of contract
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}    
    input_data = np.array([[tenure, monthly_charges, contract_map[contract]]])
    prediction = model_t3.predict_proba(input_data)
    churn_probability = prediction[0][1]
    churn_percent = f"{churn_probability:.0%}"
    #st.success(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")
    
    #add summary to the top of the page
    st.subheader("Churn Prevention & Plan Recommendation")
    st.write(" ")
    #st.subheader(f"Customer ID: {selected_customer_id}")
    selectedCustomer = f"Customer ID: {selected_customer_id}"
    #st.metric(label="Churn Risk", value=churn_percent) # Get value of delta when runing historic models, delta="-3% from last month")    

    styled_metric("Churn Risk", churn_percent, selectedCustomer )
    
    #factors of churn
    # Create a waterfall plot for that customer
    st.markdown("<div style='margin-top: 20px'></div>", unsafe_allow_html=True)
    if st.toggle("Show churn factor waterfall"):
        st.markdown("### Factors of Churn")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[i], show=False)
        st.pyplot(fig)

    #fig, ax = plt.subplots()
    #shap.plots.waterfall(shap_values[i], show=False)
    #st.pyplot(fig)
    #Below we can display the customers feature in a table form
    #st.write("Customer Features:")
    #st.dataframe(X.iloc[i:i+1])   
    
    #Recommend Plan
    
    st.markdown("### 🛠️ Recommended Retention Actions")
    st.info(recommend_action(churn_probability))
    
    # dropdown to allow manual override
    available_plans = ["Basic", "Premium", "Family", "Enterprise"]
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
    new_prob = get_newProb(override, tenure, customer_contract, model_t3)
    #st.write(new_prob)
    st.markdown(f"**Estimated Churn Probability for {override} Plan:** {new_prob:.2%}")    
    
    #customer info display
    with st.expander("Customer History", expanded=False):
        st.write(df.iloc[i])    
    #check recommandation outcome
    #st.markdown("#### Recommendation Outcome")
    #st.radio("Was the recommendation accepted?", ["Yes", "No", "Pending"])
    #st.text_area("Agent Notes")
    #st.button("Submit Feedback")
    
    # Display plan data on the side bar
    plan_prices = {
        "Basic": "$25",
        "Standard": "$55",
        "Premium": "$85",
        "Family": "$115",
        "Enterprise": "$145"
    }

    
    
       

       
    
    ###################Chat box#############################


def summarize_customer(customer):
    churn_prob = customer["churn_probability"]
    top_factors = customer["top_feature"]
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
        f"📌 **Key Factors**: {factors}\n\n"
        f"💡 **Suggestion**: Consider offering the **{plan_suggestion}** plan to improve retention."
    )

def generate_response(question, data):
    question = question.lower()
    churn_prob = data.get("churn_probability", 0.0)
    top_features = data.get("top_feature", [])
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
        return (
            f"The top factors influencing churn are: {', '.join(top_features)}. "
            f"These features have the highest SHAP impact on the prediction."
        )

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

#Assistant churnMate ############################################################

#get new prob of the overrided plan
def get_newProb(val, tenure, contract, model_t3):
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
    prediction = model_t3.predict_proba(input_data)
    new_churn_probability = prediction[0][1]
    return new_churn_probability


def recommend_action(prob):
    if prob >= 0.70:
        return "Offer personalized retention plan with discounts or loyalty perks"
    elif prob >= 0.40:
        return "Send targeted engagement emails and check-in via customer service"
    else:
        return "Maintain relationship with regular updates and appreciation messages"

def styled_metric(label, value, customerID): 
    value = float(value.strip("%")) 
    if value < 25:
        color = "green"
    elif value < 50:
        color = "orange"
    else:
        color = "red"

    st.markdown(
        f"""
        <div style='
            background-color: rgba(0,0,0,0.03);
            padding: 10px;
            border-radius: 8px;
            border-left: 5px solid {color};
            gap: 20px;
        '>
            <span style='font-size: 24px; font-weight: bold; margin-right: 20px;'>{customerID}</span><br>
            <span style='font-size: 24px; font-weight: bold; margin-right: 20px;'>{label}</span>
            <span style='font-size: 28px; color: {color}; font-weight: bold;'>{value:.0f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )

