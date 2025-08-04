import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
import plotly.express as px


def runHome():    
    with open("model_top3.pkl", "rb") as file:
        model = pickle.load(file)
    st.subheader("Churn Prediction Model")
    st.subheader("UTS P1 - 2025")
    st.write("Authors: Murray Atkin, Ibrahim Kassem, Bradley Moore, Preeti Sowrab")
    tenure = st.number_input("Tenure", min_value=0, step=1, format="%d") #get tenure as a whole number
    monthly_charges = st.number_input("Monthly Charges") #get monthly charges as a decimal number
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    # encode categorical input of contract
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    input_data = np.array([[tenure, monthly_charges, contract_map[contract_type]]])
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")
    import modelsPerformance
    modelsPerformance.run()    

def run_retention():
    #get the prediction model
    with open("model_all.pkl", "rb") as f:
        model = pickle.load(f)
    #import the dataset
    X = pd.read_csv("encoded-dataset.csv")
    #explainer = shap.Explainer(model, X)
    #shap_values = explainer(X)
    
    # Load your dataset to extract customer ids
    df = pd.read_csv("Customer-Churn-dataset.csv")
    df = df[df['Churn'] == 'No']
    
    #print model's feature order
    #st.write(model.feature_names_in_)
    
    # tenure group in 3 categories, New - Loyal - Long-term
    def tenure_group(tenure):
        if tenure <= 12:
            return 'New'
        elif 12 < tenure <= 24:
            return 'Loyal'
        else:
            return 'Long-term'
    
    df['tenure_group'] = df['tenure'].apply(tenure_group)
    
    # Recreate MonthlyCharges_Tenure if it was a product
    df["MonthlyCharges_Tenure"] = df["MonthlyCharges"] * df["tenure"]
    
    #remove unwated features
    cols_to_drop = ["customerID", "tenure", "Churn"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    #st.write("🔎 Columns currently in df:")
    #st.write(df.columns.tolist())
    
    def align_features(df, model):
        return df[model.feature_names_in_]
    
    df_aligned = align_features(df, model)
    
    #encode the dataset
    df_encoded = df_aligned.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        # Exclude 'TotalCharges' for now as it seems to have non-numeric values that need handling
        if col not in ['customerID', 'TotalCharges']:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    
    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    
    # Drop rows with NaN values created by the conversion
    df_encoded.dropna(inplace=True) 
    
    # Predict probabilities
    churn_probs = model.predict_proba(df_encoded)[:, 1]   
    
    
    # Add the prediction back into your DataFrame
    df_encoded["churn_probability"] = churn_probs
    
    #Set the risk thresholds via streamlit slider for a dynamic input
    st.sidebar.header("Set Risk Thresholds")
    
    high_threshold = st.sidebar.slider("High Risk Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    medium_threshold = st.sidebar.slider("Medium Risk Threshold", min_value=0.0, max_value=high_threshold, value=0.4, step=0.01)
    
    #set risk tires and generat tags
    def categorize_risk(prob):
        if prob >= high_threshold:
            return "High Risk 🚨"
        elif prob >= medium_threshold:
            return "Medium Risk ⚠️"
        else:
            return "Low Risk ✅"
            
    df_encoded["risk_category"] = df_encoded["churn_probability"].apply(categorize_risk)
    
    #visualize in streamlit
    import plotly.express as px
    
    risk_counts = df_encoded["risk_category"].value_counts().reset_index()
    fig = px.pie(risk_counts, names="risk_category", values="count", title="Churn Risk Distribution")
    st.plotly_chart(fig)
    
    risk_counts = df_encoded["risk_category"].value_counts()
    
    st.subheader("Risk Tier Distribution")
    
    for tier in ["High Risk 🚨", "Medium Risk ⚠️", "Low Risk ✅"]:
        count = risk_counts.get(tier, 0)
        percent = count / len(df)
        st.write(f"{tier}: {count} customers")
        st.progress(percent)



def run_customerService():
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
    st.title("Churn Prevention & Plan Recommendation App")
    st.subheader(f"Customer ID: {selected_customer_id}")
    st.metric(label="Churn Risk", value=churn_percent, delta="-3% from last month")
    
    #factors of churn
    # Create a waterfall plot for that customer
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[i], show=False)
    st.pyplot(fig)
    #Below we can display the customers feature in a table form
    #st.write("Customer Features:")
    #st.dataframe(X.iloc[i:i+1])
    
    
    #Recommend Plan
    
    st.markdown("### 🛠️ Recommended Retention Actions")
    st.info(recommend_action(churn_probability))
    
    # Optional dropdown to allow manual override
    available_plans = ["Basic", "Premium", "Family", "Enterprise"]
    override = st.selectbox("Override Plan Suggestion", options=available_plans)
      
    #customer info display
    with st.expander("Customer History", expanded=False):
        st.write(df.iloc[i])
    
    #check recommandation outcome
    st.markdown("#### Recommendation Outcome")
    st.radio("Was the recommendation accepted?", ["Yes", "No", "Pending"])
    st.text_area("Agent Notes")
    st.button("Submit Feedback")

def recommend_action(prob):
    if prob >= 0.70:
        return "Offer personalized retention plan with discounts or loyalty perks"
    elif prob >= 0.40:
        return "Send targeted engagement emails and check-in via customer service"
    else:
        return "Maintain relationship with regular updates and appreciation messages"
    

#add menu tab to navigate between pages
selected = option_menu(
    menu_title=None,
    options=["Home", "Analysis", "Cust Service", "Retention"],
    icons=["house", "bar-chart", "person-lines-fill", "shield-check"],
    orientation="horizontal"
)


if selected == "Home":
    st.title("🏠 Home Page")
    runHome()
elif selected == "Analysis":
    st.title("📊 Analysis Page")
    import churn_analysis
    churn_analysis.run()
elif selected == "Cust Service":
    st.title("👤 Customer Service Page")
    run_customerService()
elif selected == "Retention":
    st.title("🛡️ Retention Page")
    run_retention()




















