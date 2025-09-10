import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from settings import DATA_PATH

def run():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Load feature store
    with open("feature_store/logistic_ffs.json", "r") as f:
        feature_data = json.load(f)
    feature_store = feature_data.get("features", [])

    # Convert column to numeric and fill missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.rename(columns={"tenure": "Months"}, inplace=True)
    median_value = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_value)

    # Feature engineering
    def tenure_group(tenure):
        if tenure <= 12:
            return 'New'
        elif 12 < tenure <= 24:
            return 'Loyal'
        else:
            return 'Long-term'

    df['loyalty_band'] = df['Months'].apply(tenure_group)
    df['charge_velocity'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-5)

    # Encode categorical variables
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        if col not in ['customerID', 'TotalCharges']:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')

    # Contract stability scoring
    def score(row):
        score = 0
        if row['Contract'] == 'Month-to-month':
            score += 2
        elif row['Contract'] == 'One year':
            score += 1
        # Two year = 0

        if row['PaymentMethod'] in ['Electronic check', 'Mailed check']:
            score += 1

        if row['PaperlessBilling'] == 'Yes':
            score -= 1

        return score

    df_encoded['contract_stability'] = df.apply(score, axis=1)

    # Clean and filter
    df_encoded.dropna(inplace=True)
    df_encoded.columns = df_encoded.columns.str.strip()
    df_encoded.drop('customerID', axis=1, inplace=True)
    df_cleaned = df_encoded[feature_store]

    return df_cleaned
