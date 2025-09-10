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
    #with open(MODEL_PATH_T21, "rb") as f:
     #   model = pickle.load(f)

    # GitHub API URL for the file
    url = "https://api.github.com/repos/IbbyKazzi/customerChurn/contents/" + MODEL_PATH_T21
    
    # Headers (token optional for public repos)
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": st.secrets["GITHUB_TOKEN"]  # Remove if public
    }

    # Step 1: Get metadata and encoded content
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    # Step 2: Decode base64 content
    encoded_content = data["content"]
    decoded_bytes = base64.b64decode(encoded_content)
    
    # Step 3: Unpickle the model
    model = pickle.loads(decoded_bytes)
    
    #load the dataset
    import load_dataset
    df_encoded = load_dataset.run()  #this function returnes encoded dataset with 22 features  
    X = df_encoded.drop(['Churn'], axis=1)   
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # Load your dataset to extract customer ids
    df = pd.read_csv(DATA_PATH)
    df_filtered = df[df['Churn'] == 'No']
    df.rename(columns={"tenure": "Months"}, inplace=True)    
    # Extract unique customer IDs and reset index
    customer_ids_df = df_filtered['customerID'].reset_index()
    
    # Create a mapping from index to customer ID
    index_to_id = dict(zip(customer_ids_df['index'], customer_ids_df['customerID']))
    
    # Sidebar selection using index
    selected_index = st.sidebar.selectbox("Select Customer Index", options=list(index_to_id.keys()))
    
    # Retrieve the corresponding customer ID
    selected_customer_id = index_to_id[selected_index]
    
    # Now get the original index from the df
    i = customer_ids_df[customer_ids_df['customerID'] == selected_customer_id]['index'].values[0]
    
    #get selected customer's tenure,monthly charge and contract and use our prediction model to check churn possibility
    tenure = df.iloc[i]["Months"]
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
    st.session_state.prev_customer_id = selected_customer_id
    #st.metric(label="Churn Risk", value=churn_percent) # Get value of delta when runing historic models, delta="-3% from last month")    

    styled_metric("Churn Risk", churn_percent, selectedCustomer )

    #get insight from the ChurnMate assistant
    customer = {
        "name": selected_customer_id,
        "churn_probability": churn_probability,
        "monthlyCharges": monthly_charges,
        "tenure": tenure,
        "index": i
    } 
    
    # Display plan data on the side bar
    
    st.sidebar.header("📦 Available Plans")   
    plan_prices = {
        "Basic": "$25",
        "Standard": "$55",
        "Premium": "$85",
        "Family": "$115",
        "Enterprise": "$145"
    }

    # Display each plan
    for plan, price in plan_prices.items():
        st.sidebar.write(f"**{plan}**: {price}") 
      
    if selected_customer_id != st.session_state.prev_customer_id:
        st.session_state.prev_customer_id = selected_customer_id
        # 🔁 Call your function here
        def on_customer_change(customer_id):
            import customerServiceAssistance
            customerServiceAssistance.run(customer, shap_values[i], X, contract_map, df, True)
        on_customer_change(selected_customer_id) 
    else:        
        import customerServiceAssistance
        customerServiceAssistance.run(customer, shap_values[i], X, contract_map, df, False)
    

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

