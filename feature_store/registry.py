import importlib
import streamlit as st
import os
from datetime import datetime
import pytz
import json
from github import Github
import base64

FEATURES = ['loyalty_band', 'charge_velocity', 'contract_stability']



def saveToGit(name, model_meta, model, model_filename):
    try:
        REPO_NAME = "IbbyKazzi/customerChurn"
        token = st.secrets["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_NAME)

        # --- Selected Features ---
        local_dir = "feature_store"
        file_path = os.path.join(local_dir, f"{name}.json")
        with open(file_path, "r") as f:
            content = f.read()
        github_path = f"feature_store/{name}.json"

        # --- Encode model registry JSON ---
        encoded_meta = base64.b64encode(model_meta.encode()).decode()

        # --- Encode model binary ---
        with open(model_filename, "rb") as f:
            encoded_model = base64.b64encode(f.read()).decode()

        # --- Upload logic ---
        try:
            # Update selected features
            existing_file = repo.get_contents(github_path)
            repo.update_file(
                path=github_path,
                message="🔄 Update selected features",
                content=content,
                sha=existing_file.sha
            )
            st.success(f"📤 Updated file on GitHub: {github_path}")

            # Update model registry
            meta_path = "models/model_metadata.json"
            existing_file = repo.get_contents(meta_path)
            repo.update_file(
                path=meta_path,
                message="🔄 Update model register",
                content=encoded_meta,
                sha=existing_file.sha
            )

            # Update model file
            model_path = "models/" + os.path.basename(model_filename)
            existing_file = repo.get_contents(model_path)
            repo.update_file(
                path=model_path,
                message="🔄 Update best model",
                content=encoded_model,
                sha=existing_file.sha
            )

        except Exception:
            # Create selected features
            repo.create_file(
                path=github_path,
                message="🆕 Add selected features",
                content=content
            )
            st.success(f"📤 Created file on GitHub: {github_path}")

            # Create model registry
            meta_path = "models/model_metadata.json"
            repo.create_file(
                path=meta_path,
                message="🆕 Add model register",
                content=encoded_meta
            )

            # Create model file
            model_path = "models/" + os.path.basename(model_filename)
            repo.create_file(
                path=model_path,
                message="🆕 Add best model",
                content=encoded_model
            )

    except Exception as e:
        st.error(f"❌ GitHub upload failed: {e}")
        

def get_features(df, selected=FEATURES):
    result = df[['customerID']].copy()
    for feat in selected:
        module = importlib.import_module(f'feature_store.definitions.{feat}')
        func = getattr(module, feat)
        result = result.merge(func(df), on='customerID')
    return result

def get_sydney_timestamp():
    tz_sydney = pytz.timezone("Australia/Sydney")
    return datetime.now(tz_sydney).strftime("%Y-%m-%d %H:%M:%S %Z")

def save_selected_features(name, features):  
    # Prepare payload
    payload = features

    # Local save
    local_dir = "feature_store"
    os.makedirs(local_dir, exist_ok=True)
    file_path = os.path.join(local_dir, f"{name}.json")

    try:
        with open(file_path, "w") as f:
            json.dump(payload, f)
        st.success(f"✅ Saved locally to: {file_path}")
        
    except Exception as e:
        st.error(f"❌ Local save failed: {e}")
        return

def load_selected_features(name):
    import json
    with open(f"feature_store/{name}.json", "r") as f:
        return json.load(f)
