import pickle
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from settings import DATA_PATH

def run():
  #get the dataset
    df = pd.read_csv(DATA_PATH)
  
    # Convert column to numeric (in case it's still object type) and fill in missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # rename tenure column with month to avoid confusion
    df.rename(columns={"tenure": "Months"}, inplace=True)
    # Fill NaNs with median
    median_value = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_value)

    #Features Engineering
    # tenure group in 3 categories, New - Loyal - Long-term
    def tenure_group(tenure):
        if tenure <= 12:
            return 'New'
        elif 12 < tenure <= 24:
            return 'Loyal'
        else:
            return 'Long-term'
    
    df['loyalty_band'] = df['Months'].apply(tenure_group)

    #add another feature
    df['charge_velocity'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-5)
    
    # Encode categorical variables
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        # Exclude 'TotalCharges' for now as it seems to have non-numeric values that need handling
        if col not in ['customerID', 'TotalCharges']:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    
    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    def score(row):
        score = 0
        if row['Contract'] == 'Month-to-month':
            score += 2
        elif row['Contract'] == 'One year':
            score += 1
        # Two year = 0 (most stable)

        if row['PaymentMethod'] in ['Electronic check', 'Mailed check']:
            score += 1  # Less automated = more churn-prone

        if row['PaperlessBilling'] == 'Yes':
            score -= 1  # Slightly more stable

        return score

    df['contract_stability'] = df.apply(score, axis=1)
    
    # Drop rows with NaN values created by the conversion
    df_encoded.dropna(inplace=True)
    df_encoded.columns = df_encoded.columns.str.strip()
    df_encoded.drop('customerID', axis=1, inplace=True)
    return df_encoded

