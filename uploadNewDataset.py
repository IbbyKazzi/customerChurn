import streamlit as st
import pandas as pd
import os
from settings import DATA_PATH

# Upload a new dataset
def run():
  uploaded_file = st.file_uploader("📂 Upload a new dataset", type=["csv"])
  if "overwrite_done" not in st.session_state:
      st.session_state["overwrite_done"] = False
  if uploaded_file is not None:
      df = pd.read_csv(uploaded_file)
      st.success("CSV file loaded successfully!")
      st.dataframe(df.head())
      # Show basic statistics
      st.subheader("📈 Dataset Overview")
      st.write(f"**Number of rows:** {df.shape[0]}")
      st.write(f"**Number of columns:** {df.shape[1]}")
      st.write(f"**Column names:** {list(df.columns)}")
      st.write(f"**Missing values:** {df.isnull().sum().sum()}")
      st.write(f"**Duplicate rows:** {df.duplicated().sum()}")
      st.subheader("🔍 Descriptive Statistics")
      st.dataframe(df.describe())

  # Confirm overwrite
  if os.path.exists(DATA_PATH) and not st.session_state.overwrite_done:
    st.warning("⚠️ A file already exists at the save location.")
    if st.button("🔄 Commit Data Change"):
      df.to_csv(DATA_PATH, index=False)
      st.session_state.overwrite_done = True
      st.success(f"File overwritten and saved to: {DATA_PATH}")
      saveToGithub(df)
      st.rerun()
    elif not os.path.exists(DATA_PATH):
      if st.button("🔄 Commit Data Change"):
        df.to_csv(DATA_PATH, index=False)
        st.session_state.overwrite_done = True
        st.success(f"File saved to: {DATA_PATH}")
        saveToGithub(df)
        st.rerun()
      else:
        if not st.session_state.overwrite_done:
          st.info("Please upload a CSV file to proceed.")

def saveToGithub(df):
    from github import Github
    import pandas as pd
    import io
        
    # GitHub credentials    
    repo_name = "IbbyKazzi/customerChurn"
    file_path = "data/customer_churn_data.csv"
    commit_message = "Update churn data to Github"

   
    # Authenticate
    token = st.secrets["GITHUB_TOKEN"]    
    g = Github(token)     
    repo = g.get_repo(repo_name)
    
    # Load updated DataFrame        
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
        
    # Get current file content
    file = repo.get_contents(file_path, ref="main") 
    repo.update_file(
        path=file.path,
        message=commit_message,
        content=csv_buffer.getvalue(),
        sha=file.sha,
        branch="main"  # must match the ref above
    )
