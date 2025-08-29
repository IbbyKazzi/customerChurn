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

    #get current model accuracy value and display it in the st sidebar
    current_model = df[df["active"] == True].iloc[0]  
    st.sidebar.write(f"**Current Modle**")
    st.sidebar.write(f"Version: {current_model['version']}")
    st.sidebar.write("ROC AUC: " + f"**{current_model['roc_auc']:.0%}**")    
    st.sidebar.write(f"Activation Date: {current_model['date']}")
    
    #set monitoring threshold
    st.sidebar.header("Model Monitoring")
    auc_threshold = st.sidebar.slider("Minimum AUC Threshold", 0.5, 1.0, 0.85, step=0.01)

    auc_history_df = df[["date","roc_auc"]] #get model's aucs
    st.line_chart(auc_history_df.set_index("date")["roc_auc"])

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

    # Initialize session state flag
    if "run_pipeline" not in st.session_state:
        st.session_state.run_pipeline = False
    
    # Button to trigger pipeline
    if st.button("🔄 Run Pipeline"):
        st.session_state.run_pipeline = True
    
    # Run pipeline if triggered
    if st.session_state.run_pipeline:
        with st.spinner("Running pipeline..."):
            import automated_pipeline as ap
            X_df, y, (X_train_full, X_test_full, y_train, y_test) = ap.load_and_preprocess(DATA_PATH)
    
            selected_features, ffs_scores = ap.forward_feature_selection(
                pd.DataFrame(X_train_full, columns=X_df.columns), y_train
            )
    
            from feature_store.registry import save_selected_features
            save_selected_features("logistic_ffs", selected_features)
    
            X_train = pd.DataFrame(X_train_full, columns=X_df.columns)[selected_features]
            X_test = pd.DataFrame(X_test_full, columns=X_df.columns)[selected_features]
    
            models = ap.train_models(X_train, y_train)
            model_scores = ap.evaluate_models(models, X_test, y_test)
    
            st.success("✅ Pipeline completed!")
    
            scores_df = pd.DataFrame(model_scores).T.reset_index().rename(columns={"index": "Model"})
            st.subheader("📋 Model Metrics")
            st.dataframe(scores_df)
    
            ap.select_best_model(model_scores, metric="AUC")
    
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
    
        # ✅ Reset the flag so it doesn't run again on next rerun
        st.session_state.run_pipeline = False

    


