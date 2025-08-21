import streamlit as st
import pandas as pd
import json
from settings import METADATA_PATH, DATA_PATH

def show_model_history(path=METADATA_PATH):
    #st.header("🧠 Model Metadata Dashboard")

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


    if st.button("Run Pipeline"):
        with st.spinner("Running pipeline..."):
            import automated_pipeline as ap
            model_scores = ap.load_and_preprocess(DATA_PATH)
            st.success("✅ Pipeline completed!")
    
            # Display metrics
            model_scores = ap.
            scores_df = pd.DataFrame(model_scores).T.reset_index().rename(columns={"index": "Model"})
            st.subheader("📋 Model Metrics")
            st.dataframe(scores_df.style.format("{:.2f}"))
    
            # Metric comparison
            selected_metric = st.selectbox("Compare metric", scores_df.columns[1:])
            st.bar_chart(scores_df.set_index("Model")[selected_metric])

