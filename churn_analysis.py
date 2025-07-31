#define 5 monthly plan tires
import pandas as pd

results = []


# Sample data
data = {
    "MonthlyCharges": [25, 55, 85, 115, 145],
    "tenure":[1,1,1,1,1],
    "Contract":[0,0,0,0,0]
}
# Create the DataFrame
df = pd.DataFrame(data)

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

churn_by_plan["ContractType"] = "Month-to-Month" 
results.append(churn_by_plan)


#visualize using streamlit
#import streamlit as st
#import plotly.express as px

#st.title("📈 Churn Rate by Monthly Plan")
#st.subheader("    Month-to-Month Contract")

#fig = px.bar(churn_by_plan, x="Plan", y="Predicted Churn Rate",
#             title="Which Plan Drives Churn?",
#             color="Predicted Churn Rate", color_continuous_scale="OrRd"
#)

#st.plotly_chart(fig)

# Sample data yearly contract
data = {
    "MonthlyCharges": [25, 55, 85, 115, 145],
    "tenure":[1,1,1,1,1],
    "Contract":[1,1,1,1,1]
}
# Create the DataFrame
df = pd.DataFrame(data)

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

churn_by_plan["ContractType"] = "Yearly" 
results.append(churn_by_plan)

#visualize using streamlit
#import streamlit as st
#import plotly.express as px

#st.title("📈 Churn Rate by Monthly Plan")
#st.subheader("    Yearly Contract")
#fig = px.bar(churn_by_plan, x="Plan", y="Predicted Churn Rate",
#             title="Which Plan Drives Churn?",
#             color="Predicted Churn Rate", color_continuous_scale="OrRd"
#)

#st.plotly_chart(fig)



# Sample data 2 years contract
data = {
    "MonthlyCharges": [25, 55, 85, 115, 145],
    "tenure":[1,1,1,1,1],
    "Contract":[2,2,2,2,2]
}
# Create the DataFrame
df = pd.DataFrame(data)

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

churn_by_plan["ContractType"] = "2-Year" 
results.append(churn_by_plan)

#visualize using streamlit
import streamlit as st
#import plotly.express as px

#st.title("📈 Churn Rate by Monthly Plan")
#st.subheader("    2 Years Contract")
#fig = px.bar(churn_by_plan, x="Plan", y="Predicted Churn Rate",
#             title="Which Plan Drives Churn?",
#             color="Predicted Churn Rate", color_continuous_scale="OrRd"
#)

#st.plotly_chart(fig)


matrix_df = pd.concat(results)
matrix_df = matrix_df.pivot(index="Plan", columns="ContractType", values="Predicted Churn Rate").reset_index()

#visualise in streamlit
#st.title("📋 Churn Rate Matrix by Plan and Contract Type")
#st.dataframe(matrix_df.style.format("{:.2%}"))

import plotly.express as px

heatmap_df = matrix_df.set_index("Plan")
fig = px.imshow(heatmap_df,
                labels=dict(x="Contract Type", y="Plan", color="Churn Rate"),
                x=heatmap_df.columns,
                y=heatmap_df.index,
                color_continuous_scale="Reds",
                text_auto=True,
                width=900,   # Increase width
    height=600   # Increase height
)

st.plotly_chart(fig)
