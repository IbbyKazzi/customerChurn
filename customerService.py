import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go



#get the prediction model
with open("model_all.pkl", "rb") as f:
    model = pickle.load(f)
#import the dataset
X = pd.read_csv("encoded-dataset.csv")
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Load your dataset to extract customer ids
df = pd.read_csv("Customer-Churn-dataset.csv")
# Extract unique customer IDs
customer_ids = df["customerID"].unique().tolist()


#set the page menu  Customer-Churn-dataset.csv
st.sidebar.header("Customer Filter")
selected_customer_id = st.sidebar.selectbox("Enter Customer ID", options=customer_ids)
contract_type = st.sidebar.selectbox("Contract Type", options=["Monthly", "One Year", "Two Year"])
# Choose the customer index
i = customer_ids.index(selected_customer_id)

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
prediction = model_t3.predict(input_data)
risk_score = prediction[0][1]
#st.success(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")

#add summary to the top of the page
st.title("Churn Prevention & Plan Recommendation App")
st.subheader(f"Customer ID: {selected_customer_id}")
st.metric(label="Churn Risk", value=f"{risk_score:.0%}", delta="-3% from last month")



#factors of churn
# Create a waterfall plot for that customer
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[i], show=False)
st.pyplot(fig)
#Below we can display the customers feature in a table form
#st.write("Customer Features:")
#st.dataframe(X.iloc[i:i+1])


#Recommend Plan

st.markdown("### Recommended Retention Plan")
st.info("Switch to 'SecureNet Plus': 24-month contract, free upgrade, loyalty rewards.")

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
  
