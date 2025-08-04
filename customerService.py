import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def run():
    #get the prediction model
    with open("model_all.pkl", "rb") as f:
        model = pickle.load(f)
    #import the dataset
    X = pd.read_csv("encoded-dataset.csv")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # Load your dataset to extract customer ids
    df = pd.read_csv("Customer-Churn-dataset.csv")
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
    with open("model_top3.pkl", "rb") as f:
        model_t3 = pickle.load(f)
    # encode categorical input of contract
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}    
    input_data = np.array([[tenure, monthly_charges, contract_map[contract]]])
    prediction = model_t3.predict_proba(input_data)
    churn_probability = prediction[0][1]
    churn_percent = f"{churn_probability:.0%}"
    #st.success(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")
    
    #add summary to the top of the page
    st.subheader("Churn Prevention & Plan Recommendation App")
    st.subheader(f"Customer ID: {selected_customer_id}")
    #st.metric(label="Churn Risk", value=churn_percent) # Get value of delta when runing historic models, delta="-3% from last month")    

    styled_metric("Churn Risk", churn_percent )
    
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
    st.markdown("#### Recommendation Outcome")
    st.radio("Was the recommendation accepted?", ["Yes", "No", "Pending"])
    st.text_area("Agent Notes")
    st.button("Submit Feedback")

#get new prob of the overrided plan
def get_newProb(val, tenure, contract, model_t3):
    with open("model_top3.pkl", "rb") as f:
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

def styled_metric(label, value): 
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
            <span style='font-size: 24px; font-weight: bold; margin-right: 20px;'>{label}</span>
            <span style='font-size: 28px; color: {color}; font-weight: bold;'>{value:.0f}%</span>

        </div>
        """,
        unsafe_allow_html=True
    )

