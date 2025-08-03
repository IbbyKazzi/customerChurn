import streamlit as st
import numpy as np
import pickle
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
import streamlit as st
import streamlit as st
import plotly.express as px


def runHome():    
    with open("model_top3.pkl", "rb") as file:
        model = pickle.load(file)
    st.title("Churn Prediction Model")
    st.title("UTS P1 - 2025")
    st.subheader("Authors: Murray Atkin, Ibrahim Kassem, Bradley Moore, Preeti Sowrab")
    tenure = st.number_input("Tenure", min_value=0, step=1, format="%d") #get tenure as a whole number
    monthly_charges = st.number_input("Monthly Charges") #get monthly charges as a decimal number
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    # encode categorical input of contract
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    input_data = np.array([[tenure, monthly_charges, contract_map[contract_type]]])
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")
    
    #if st.button("Go to Churn Analysis 📊"):
    #    st.switch_page("churn_analysis.py")  
    
    #github_url = "https://customerchurn-utsp1-analysis.streamlit.app/"
    
    #st.markdown(
    #    f"""
    #    <a href="{github_url}" target="_blank">
    #        <button style="background-color:#1f77b4; color:white; padding:10px 24px; font-size:16px; border:none; border-radius:5px;">
    #            🚀 Go to Churn Analysis 📊
    #        </button>
    #    </a>
    #    """,
    #    unsafe_allow_html=True
    #)

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

def run_analysis():
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
    
    st.title("📈 Churn Rate by Monthly Plan")
    st.subheader("    Month-to-Month Contract")
    
    fig = px.bar(churn_by_plan, x="Plan", y="Predicted Churn Rate",
                 title="Which Plan Drives Churn?",
                 color="Predicted Churn Rate", color_continuous_scale="OrRd",
                 category_orders={"Plan": ["Basic", "Standard", "Premium", "Family", "Enterprise"]}
    )    
    st.plotly_chart(fig)
    
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
    
    st.title("📈 Churn Rate by Monthly Plan")
    st.subheader("    Yearly Contract")
    fig = px.bar(churn_by_plan, x="Plan", y="Predicted Churn Rate",
                 title="Which Plan Drives Churn?",
                 color="Predicted Churn Rate", color_continuous_scale="OrRd",
                 category_orders={"Plan": ["Basic", "Standard", "Premium", "Family", "Enterprise"]}
    )
    
    st.plotly_chart(fig)  
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
    
    st.title("📈 Churn Rate by Monthly Plan")
    st.subheader("    2 Years Contract")
    fig = px.bar(churn_by_plan, x="Plan", y="Predicted Churn Rate",
                 title="Which Plan Drives Churn?",
                 color="Predicted Churn Rate", color_continuous_scale="OrRd",
                 category_orders={"Plan": ["Basic", "Standard", "Premium", "Family", "Enterprise"]}
    )    
    st.plotly_chart(fig)    
    
    matrix_df = pd.concat(results)
    matrix_df = matrix_df.pivot(index="Plan", columns="ContractType", values="Predicted Churn Rate").reset_index() 
   
    # Define the desired order
    plan_order = ["Basic", "Standard", "Premium", "Family", "Enterprise"]
    
    # Convert 'Plan' to a categorical type with the specified order
    matrix_df["Plan"] = pd.Categorical(matrix_df["Plan"], categories=plan_order, ordered=True)
    
    # Sort the DataFrame by the ordered 'Plan'
    matrix_df = matrix_df.sort_values("Plan")
    
    # Visualise in Streamlit
    st.title("📋 Churn Rate Matrix")
    st.subheader("   By Plan and Contract Type")
    st.dataframe(matrix_df, hide_index=True)
        
    st.subheader("Heat-map visual")
    heatmap_df = matrix_df.set_index("Plan")
    fig = px.imshow(heatmap_df,
                    labels=dict(x="Contract Type", y="Plan", color="Churn Rate"),
                    x=heatmap_df.columns,
                    y=heatmap_df.index,
                    color_continuous_scale="Reds",
                    text_auto=True,
                    width=900,   # Increase width
        height=600,   # Increase height
                    
    )    
    st.plotly_chart(fig)   
    

selected = option_menu(
    menu_title=None,
    options=["Home", "Analysis", "Cust Service", "Retention"],
    icons=["house", "bar-chart", "person-lines-fill","shield-check" ],
    orientation="horizontal"
)

if selected == "Home":
    st.title("🏠 Home Page")
    runHome()
elif selected == "Analysis":
    st.title("📊 Analysis Page")
elif selected == "Cust Service":
    st.title("👤 Customer Service Page")
elif selected == "Retention":
    st.title("🛡️ Retention Page")
    run_retention()










