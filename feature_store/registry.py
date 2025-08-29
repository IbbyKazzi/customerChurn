import importlib
import streamlit as st

FEATURES = ['loyalty_band', 'charge_velocity']

def get_features(df, selected=FEATURES):
    result = df[['customerID']].copy()
    for feat in selected:
        module = importlib.import_module(f'feature_store.definitions.{feat}')
        func = getattr(module, feat)
        result = result.merge(func(df), on='customerID')
    return result

def save_selected_features(name, features):
    import json 
    file_path = f"feature_store/{name}.json"
    os.makedirs("feature_store", exist_ok=True)
    try:        
        with open(file_path, "w") as f:
            json.dump(features, f)
            st.write(features)
            st.success(f"Saved selected features to: {file_path}")
    except Exception as e:
        st.error(f"Error saving features: {e}")

    


def load_selected_features(name):
    import json
    with open(f"feature_store/{name}.json", "r") as f:
        return json.load(f)
