import streamlit as st
import pandas as pd
import json
import plotly.express as px
from settings import METADATA_PATH, DATA_PATH

def show_model_history(path=METADATA_PATH):
    #st.header("🧠 Model Metadata Dashboard")

    # Load metadata
    with open(path, "r") as f:
        metadata = json.load(f)
    df = pd.DataFrame(metadata)

    # Display registry
    st.subheader("📋 Model Registry")
    st.dataframe(df[["version", "date", "accuracy", "roc_auc", "notes", "active"]])

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
            X_train, X_test, y_train, y_test = ap.load_and_preprocess(DATA_PATH)
            models = ap.train_models(X_train, y_train)
            model_scores = ap.evaluate_models(models, X_test, y_test)
            st.success("✅ Pipeline completed!")
        #st.write(models)
        # Display metrics
        scores_df = pd.DataFrame(model_scores).T.reset_index().rename(columns={"index": "Model"})
        st.subheader("📋 Model Metrics")
        #st.dataframe(scores_df.style.format("{:.2f}"))
        st.dataframe(scores_df)
        ap.select_best_model(model_scores, metric="AUC")
        # Metric comparison
        #selected_metric = st.selectbox("Compare metric", scores_df.columns[1:])
        #st.bar_chart(scores_df.set_index("Model")[selected_metric])

        scores_melted = scores_df.melt(id_vars="Model", var_name="Metric", value_name="Score")   
        
        fig = px.bar(
            scores_melted,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="📊 Model Performance Across Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)


