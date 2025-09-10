import importlib
import streamlit as st
import os
from datetime import datetime
import pytz
import json
from github import Github
import base64

FEATURES = ['loyalty_band', 'charge_velocity', 'contract_stability']

def saveToGit(name, model_obj, model_filename):
    try:
        REPO_NAME = "IbbyKazzi/customerChurn"
        token = st.secrets["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_NAME)

        # Read features content
        local_dir = "feature_store"
        file_path = os.path.join(local_dir, f"{name}.json")
        with open(file_path, "r") as f:
            features_content = f.read()    
        features_path = f"feature_store/{name}.json"

        # Read model register content        
        model_reg_path = "models/model_metadata.json"
        with open(model_reg_path, "r") as f:
            model_reg_content = f.read()    
        
    
        # Check if file exists in repo
        try:
            existing_file = repo.get_contents(features_path)
            response = repo.update_file(
                path=features_path,
                message="🔄 Update selected features",
                content=features_content,
                sha=existing_file.sha
            )           
    
            #save models register            
            existing_file = repo.get_contents(model_reg_path)
            response = repo.update_file(
                path=model_reg_path,
                message="🔄 Update model register",
                content=model_reg_content,
                sha=existing_file.sha
            )
            #save best models             
            with open(model_filename, "rb") as f:
                binary_content = f.read()
            
            model_path = model_filename
            
            # Check if file exists
            try:
                existing_file = repo.get_contents(model_path)
                repo.update_file(
                    path=model_path,
                    message="🔄 Update best model",
                    content=binary_content,
                    sha=existing_file.sha
                )
            except Exception:
                # If file doesn't exist, create it
                repo.create_file(
                    path=model_path,
                    message="📦 Add best model",
                    content=binary_content
                )


            st.success(f"📤 Updated file on GitHub: {github_path}")
    
        except Exception:
            existing_file = repo.get_contents(features_path)
            response = repo.update_file(
                path=features_path,
                message="🔄 Update selected features",
                content=features_content,
                sha=existing_file.sha
            )           
    
            #save models register            
            existing_file = repo.get_contents(model_reg_path)
            response = repo.update_file(
                path=model_reg_path,
                message="🔄 Update model register",
                content=model_reg_content,
                sha=existing_file.sha
            )
            #save best models             
            import base64
            with open(model_filename, "rb") as f:
                encoded_model = base64.b64encode(f.read()).decode()

            model_path = model_filename
            existing_file = repo.get_contents(model_path)
            repo.update_file(
                path=model_path,
                message="🔄 Update best model",
                content=encoded_model,
                sha=existing_file.sha
            )
            st.success(f"📤 Updated file on GitHub: {github_path}")
            
    
    
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
    # Local save
    local_dir = "feature_store"
    os.makedirs(local_dir, exist_ok=True)
    file_path = os.path.join(local_dir, f"{name}.json")

    try:
        with open(file_path, "w") as f:
            json.dump(features, f)
        st.success(f"✅ Saved locally to: {file_path}")
        
    except Exception as e:
        st.error(f"❌ Local save failed: {e}")
        return

def load_selected_features(name):
    import json
    with open(f"feature_store/{name}.json", "r") as f:
        return json.load(f)
