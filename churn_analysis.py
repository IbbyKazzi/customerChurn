#define 5 monthly plan tires
import pandas as pd
monthly_plans = {
    "Basic": 29,
    "Standard": 59,
    "Premium": 89,
    "Family": 119,
    "Enterprise": 159
}
def assign_plan(charge):
    if charge <= 39:
        return "Basic"
    elif charge <= 69:
        return "Standard"
    elif charge <= 99:
        return "Premium"
    elif charge <= 129:
        return "Family"
    else:
        return "Enterprise"

df["Plan"] = df["MonthlyCharges"].apply(assign_plan)
#use our deployed model to predict churn for the above 5 plans
import pickle

with open("model_top3.pkl", "rb") as f:
    model = pickle.load(f)

X = df[["tenure", "MonthlyCharges", "Contract"]]  # Use the right features
df["PredictedChurn"] = model.predict(X)

#analyse churn rates by plan
churn_by_plan = df.groupby("Plan")["PredictedChurn"].mean().reset_index()
churn_by_plan.columns = ["Plan", "Predicted Churn Rate"]

#visualize using streamlit
import streamlit as st
import plotly.express as px

st.title("📈 Churn Rate by Monthly Plan")

fig = px.bar(churn_by_plan, x="Plan", y="Predicted Churn Rate",
             title="Which Plan Drives Churn?",
             color="Predicted Churn Rate", color_continuous_scale="OrRd")

st.plotly_chart(fig)
