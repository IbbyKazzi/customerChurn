import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from settings import MODEL_PATH_T3, MODEL_PATH_T21, DATA_PATH
from feature_store.registry import get_features
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np


#Using FFS logic to inhance model training and ecvaluation
def forward_feature_selection(X, y, max_features=None):
    selected = []
    remaining = list(X.columns)
    best_score = 0
    scores = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    while remaining and (max_features is None or len(selected) < max_features):
        score_candidates = []
        for feature in remaining:
            trial_features = selected + [feature]
            aucs = []

            for train_idx, val_idx in cv.split(X[trial_features], y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = LogisticRegression(C=0.01, penalty='l2', solver='liblinear', max_iter=1000)
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_val)[:, 1]
                aucs.append(roc_auc_score(y_val, y_prob))

            mean_auc = np.mean(aucs)
            score_candidates.append((mean_auc, feature))
        #st.write("Candidate scores this round:", score_candidates)
        score_candidates.sort(reverse=True)
        best_new_score, best_feature = score_candidates[0]

        tolerance = 0.005  # Accept features that don't drop AUC by more than 0.5%
        if best_new_score >= best_score - tolerance:
            selected.append(best_feature)
            remaining.remove(best_feature)
            best_score = best_new_score
            scores.append(best_score)
        else:
            break
    
    return selected, scores



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

    #get features engineer
    features_df = get_features(df)
    # Identify overlapping columns (excluding the key)
    overlap = [col for col in features_df.columns if col in df.columns and col != 'customerID']
    
    # Drop them from df before merging
    df_clean = df.drop(columns=overlap)
    
    # Merge cleanly
    enriched_df = df_clean.merge(features_df, on='customerID', how='left')
    
    #feature re-name
    enriched_df.rename(columns={"tenure": "Months"}, inplace=True)
    
    # Encode categorical variables
    df_encoded = enriched_df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        # Exclude 'TotalCharges' for now as it seems to have non-numeric values that need handling
        if col not in ['customerID', 'TotalCharges']:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])   
    
    
    # Prepare data - Exclude 'tenure_group' and 'customerID' from features
    X = df_encoded.drop(['Churn', 'customerID'], axis=1)
    y = df_encoded['Churn']   
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, y, train_test_split(X, y, test_size=0.2, random_state=42)


#Model Training (logistic & Current Model)
def train_models(X_train, y_train):
    #load current model
    with open(MODEL_PATH_T21, "rb") as f:
        model_t21 = pickle.load(f)
        
    models = {
        "Trained_Model": LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000),
        "Current_Model": model_t21
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model  

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
    st.success(f"Best model: {best_model[0]} with {metric}: {best_model[1][metric]:.4f}")
    return best_model[0]
