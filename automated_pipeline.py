import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from settings import MODEL_PATH_T3, MODEL_PATH_T21, DATA_PATH

#Data Ingestion and preprocessing
def load_and_preprocess(path):
    #load dataset
    df = pd.read_csv(path)
    #clean data
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df.loc[:, 'TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Convert column to numeric (in case it's still object type)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')    
    # Fill NaNs with median
    median_value = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_value)

    #feature engineering
    df['loyalty_band'] = df['tenure'].apply(tenure_group) #Feature 1
    df['charge_velocity'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-5) # Feature 2

    #feature re-name
    df.rename(columns={"tenure": "Months"}, inplace=True)
    
    # Encode categorical variables
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        # Exclude 'TotalCharges' for now as it seems to have non-numeric values that need handling
        if col not in ['customerID', 'TotalCharges']:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    
    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    
    # Drop rows with NaN values created by the conversion
    df_encoded.dropna(inplace=True)
    
    # Prepare data - Exclude 'tenure_group' and 'customerID' from features
    X = df_encoded.drop(['Churn', 'customerID'], axis=1)
    y = df_encoded['Churn']   
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#feature engineering
def tenure_group(tenure):
    if tenure <= 12:
        return 'New'
    elif 12 < tenure <= 24:
        return 'Loyal'
    else:
        return 'Long-term'

#Model Training (logistic & Randomforest)
def train_models(X_train, y_train):
    #models = {
    #    "LogisticRegression": LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000),
    #    "RandomForest": RandomForestClassifier(n_estimators=100)
    #}
    with open(MODEL_PATH_T21, "rb") as f:
        model_t21 = pickle.load(f)
    models = {
        "Trained_Model": LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000),
        "Current_Model": model_t21
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model  # Replace with trained model

    return models

#Model evaluation
def evaluate_models(models, X_test, y_test):
    scores = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        scores[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        }

    return scores

#Select best model
def select_best_model(scores, metric="Accuracy"):
    best_model = max(scores.items(), key=lambda x: x[1][metric])
    st.write(f"Best model: {best_model[0]} with {metric}: {best_model[1][metric]:.4f}")
    return best_model[0]
