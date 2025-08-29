import importlib
import streamlit as st
import os


FEATURES = ['loyalty_band', 'charge_velocity']

def get_features(df, selected=FEATURES):
    result = df[['customerID']].copy()
    for feat in selected:
        module = importlib.import_module(f'feature_store.definitions.{feat}')
        func = getattr(module, feat)
        result = result.merge(func(df), on='customerID')
    return result

def save_selected_features(name, features):
    import os
    import json
    import streamlit as st
    from github import Github

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

    # GitHub upload
    try:
        # Replace with your actual token and repo name
        GITHUB_TOKEN = "your_personal_access_token"  # 🔐 Store securely in Streamlit secrets or env vars
        REPO_NAME = "your_username/customerChurn"    # e.g., "ibrahim/customerChurn"

        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)

        # Read file content
        with open(file_path, "r") as f:
            content = f.read()

        github_path = f"feature_store/{name}.json"

        # Check if file exists in repo
        try:
            existing_file = repo.get_contents(github_path)
            repo.update_file(
                path=github_path,
                message="🔄 Update selected features",
                content=content,
                sha=existing_file.sha
            )
            st.success(f"📤 Updated file on GitHub: {github_path}")
        except Exception:
            repo.create_file(
                path=github_path,
                message="🆕 Add selected features",
                content=content
            )
            st.success(f"📤 Created file on GitHub: {github_path}")

    except Exception as e:
        st.error(f"❌ GitHub upload failed: {e}")


def load_selected_features(name):
    import json
    with open(f"feature_store/{name}.json", "r") as f:
        return json.load(f)
