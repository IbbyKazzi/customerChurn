import streamlit as st
import pandas as pd
import json
from settings import METADATA_PATH

def show_model_history(path=METADATA_PATH):
    st.header("🧠 Model Metadata Dashboard")

    # Load metadata
    with open(path, "r") as f:
        metadata = json.load(f)
    df = pd.DataFrame(metadata)

    # Display registry
    st.subheader("📋 Model Registry")
    st.dataframe(df[["version", "date", "accuracy", "roc_auc", "notes"]])

    # Comparison section
    st.subheader("🔍 Compare Model Versions")
    versions = df["version"].tolist()
    selected_versions = st.multiselect("Select versions to compare", versions)

    if selected_versions:
        compare_df = df[df["version"].isin(selected_versions)]

        st.write("📊 Accuracy and ROC AUC")
        st.bar_chart(compare_df.set_index("version")[["accuracy", "roc_auc"]])

        # Toggles for optional views
        if st.toggle("Show Hyperparameters"):
            for _, row in compare_df.iterrows():
                st.markdown(f"**{row['version']}**")
                st.json(row["hyperparameters"])

        if st.toggle("Show Features Used"):
            for _, row in compare_df.iterrows():
                st.markdown(f"**{row['version']}**: {', '.join(row['features'])}")
