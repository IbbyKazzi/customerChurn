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
customer_id = st.sidebar.text_input("Enter Customer ID", options=customer_ids)
contract_type = st.sidebar.selectbox("Contract Type", options=["Monthly", "One Year", "Two Year"])

#add summary to the top of the page
st.title("Churn Prevention & Plan Recommendation App")
st.subheader(f"Customer ID: {customer_id}")
st.metric(label="Churn Risk", value="82%", delta="-3% from last month")

#factors of churn
"""
st.markdown("#### Key Factors Driving Churn")
feature_importance = np.abs(shap_values.values).mean(axis=0)
feature_names = X.columns

fig = go.Figure(go.Bar(
    x=feature_importance,
    y=feature_names,
    orientation='h'
))

st.plotly_chart(fig)
"""

# Choose the customer index
i = customer_ids.index(selected_customer_id)

st.write("Selected Customer ID:", selected_customer_id)
st.write("Index of Selected ID:", selected_index)


# Create a waterfall plot for that customer
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[i], show=False)
st.pyplot(fig)

st.write("Customer Features:")
st.dataframe(X.iloc[i:i+1])


#Recommend Plan

st.markdown("### Recommended Retention Plan")
st.info("Switch to 'SecureNet Plus': 24-month contract, free upgrade, loyalty rewards.")

# Optional dropdown to allow manual override
override = st.selectbox("Override Plan Suggestion", options=available_plans)
  
#customer info display
with st.expander("Customer History", expanded=False):
    st.write(customer_profile_df)

#check recommandation outcome
st.markdown("#### Recommendation Outcome")
st.radio("Was the recommendation accepted?", ["Yes", "No", "Pending"])
st.text_area("Agent Notes")
st.button("Submit Feedback")
  
