import streamlit as st
import numpy as np
#set the page menu
st.sidebar.header("Customer Filter")
customer_id = st.sidebar.text_input("Enter Customer ID")
contract_type = st.sidebar.selectbox("Contract Type", options=["Monthly", "One Year", "Two Year"])

#add summary to the top of the page
st.title("Churn Prevention & Plan Recommendation App")
st.subheader(f"Customer ID: {customer_id}")
st.metric(label="Churn Risk", value="82%", delta="-3% from last month")

#factors of churn
st.markdown("#### Key Factors Driving Churn")
st.plotly_chart(churn_driver_chart)

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
  
