import pandas as pd
import pickle

# Define plans and contract types
monthly_charges = [25, 55, 85, 115, 145]
contract_types = {0: "Month-to-Month", 1: "Yearly", 2: "2-Year"}
plan_labels = ["Basic", "Standard", "Premium", "Family", "Enterprise"]

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

# Load model
with open("model_top3.pkl", "rb") as f:
    model = pickle.load(f)

# Generate churn data across contracts
results = []
for contract_code, contract_label in contract_types.items():
    df = pd.DataFrame({
        "MonthlyCharges": monthly_charges,
        "tenure": [1]*5,
        "Contract": [contract_code]*5
    })
    df["Plan"] = df["MonthlyCharges"].apply(assign_plan)
    X = df[["tenure", "MonthlyCharges", "Contract"]]

    # check which one is better present, chrun classification or churn probability
    
    #df["PredictedChurn"] = model.predict(X)
    # Get churn probability (assuming binary classification: [No Churn, Churn])
    df["ChurnProbability"] = model.predict_proba(X)[:, 1]  # Probability of class '1' (churn)
    
    #churn_by_plan = df.groupby("Plan")["PredictedChurn"].mean().reset_index()
    #churn_by_plan.columns = ["Plan", "Predicted Churn Rate"]

    churn_by_plan = df.groupby("Plan")["ChurnProbability"].mean().reset_index()
    churn_by_plan.columns = ["Plan", "Churn Probability"]
    
    churn_by_plan["ContractType"] = contract_label
    results.append(churn_by_plan)

# Combine into matrix
matrix_df = pd.concat(results)
matrix_df["Plan"] = pd.Categorical(matrix_df["Plan"], categories=plan_labels, ordered=True)
matrix_df = matrix_df.sort_values("Plan")

pivot_df = matrix_df.pivot(index="Plan", columns="ContractType", values="Churn Probability").reset_index()

#pivot_df = matrix_df.pivot(index="Plan", columns="ContractType", values="Predicted Churn Rate").reset_index()

# Convert churn rate columns to float
for col in pivot_df.columns[1:]:  # Skip 'Plan' column
    pivot_df[col] = pd.to_numeric(pivot_df[col], errors="coerce")

#st.write(pivot_df.columns)

#streamlit UI with plan selector
import streamlit as st
import plotly.express as px

st.title("📊 Churn Rate by Plan and Contract Type")

# Dropdown to select plan
selected_plan = st.selectbox("Select a Monthly Plan", plan_labels)

# Filter data for selected plan
plan_data = matrix_df[matrix_df["Plan"] == selected_plan]

# Bar chart across contract types
fig = px.bar(
    plan_data,
    x="ContractType",
    y="Churn Probability",
    color="Churn Probability",
    color_continuous_scale="OrRd",
    title=f"Churn Rate for {selected_plan} Plan",
    labels={"ContractType": "Contract Type", "Predicted Churn Probability": "Churn Probability"}
)
st.plotly_chart(fig)



#Add heatmap with matrix view
st.subheader("📋 Full Churn Rate Matrix")
#st.dataframe(pivot_df.style.format("{:.2%}"), hide_index=True)
#st.dataframe(pivot_df, hide_index=True)

st.write(pivot_df.dtypes)
st.write(pivot_df.head())

# Build formatter dictionary for non-'Plan' columns
formatters = {
    col: "{:.2%}" for col in pivot_df.columns if col != "Plan"
}

# Apply style with targeted formatting
styled_df = pivot_df.style.format(formatters).background_gradient(cmap="Reds", axis=None)
st.dataframe(styled_df, hide_index=True)    

# Display in Streamlit
st.dataframe(styled_df, hide_index=True)

st.subheader("🧯 Heatmap View")
heatmap_df = pivot_df.set_index("Plan")
fig = px.imshow(
    heatmap_df,
    labels=dict(x="Contract Type", y="Plan", color="Churn Rate"),
    color_continuous_scale="Reds",
    text_auto=True,
    width=900,
    height=600
)
st.plotly_chart(fig)
