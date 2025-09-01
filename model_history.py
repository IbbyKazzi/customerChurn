# --- Main App ---
import streamlit as st
from settings import METADATA_PATH, DATA_PATH
import pandas as pd
import json
import plotly.express as px
from feature_store.registry import save_selected_features, saveToGit

# --- Dashboard Rendering ---
def show_model_history(path=METADATA_PATH):
    with open(path, "r") as f:
        metadata = json.load(f)
    df = pd.DataFrame(metadata)

    st.subheader("📋 Model Registry")
    st.dataframe(df[["version", "date", "accuracy", "roc_auc", "notes", "active"]])

    current_model = df[df["active"] == True].iloc[0]
    st.sidebar.write(f"**Current Model**")
    st.sidebar.write(f"Version: {current_model['version']}")
    st.sidebar.write("ROC AUC: " + f"**{current_model['roc_auc']:.0%}**")
    st.sidebar.write(f"Activation Date: {current_model['date']}")

    st.sidebar.header("Model Monitoring")
    auc_threshold = st.sidebar.slider("Minimum AUC Threshold", 0.5, 1.0, 0.85, step=0.01)

    auc_history_df = df[["date", "roc_auc"]]
    st.line_chart(auc_history_df.set_index("date")["roc_auc"])

    st.subheader("🔍 Compare Model Versions")
    versions = df["version"].tolist()
    selected_versions = st.multiselect("Select versions to compare", versions)

    if selected_versions:
        compare_df = df[df["version"].isin(selected_versions)]
        st.write("📊 Accuracy and ROC AUC")
        st.bar_chart(compare_df.set_index("version")[["accuracy", "roc_auc"]])

        if st.toggle("Show Hyperparameters"):
            for _, row in compare_df.iterrows():
                st.markdown(f"**{row['version']}**")
                st.json(row["hyperparameters"])

        if st.toggle("Show Features Used"):
            for _, row in compare_df.iterrows():
                st.markdown(f"**{row['version']}**: {', '.join(row['features'])}")
def run():
    # --- Dashboard ---
    show_model_history()
    
    # --- Session State Initialization ---
    if "run_pipeline" not in st.session_state:
        st.session_state.run_pipeline = False
    if "pipeline_ran" not in st.session_state:
        st.session_state.pipeline_ran = False
    if "save_results" not in st.session_state:
        st.session_state.save_results = ''
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = ''
    
    # --- UI Buttons ---    
    if st.button("🔄 Run Pipeline"):
        st.session_state.run_pipeline = True
        st.session_state.pipeline_ran = False  # Reset flag    
   
    # --- Pipeline Execution ---
    if st.session_state.run_pipeline and not st.session_state.pipeline_ran:
        with st.spinner("Running pipeline..."):
            import automated_pipeline as ap
            X_df, y, (X_train_full, X_test_full, y_train, y_test) = ap.load_and_preprocess(DATA_PATH)
    
            selected_features, ffs_scores = ap.forward_feature_selection(
                pd.DataFrame(X_train_full, columns=X_df.columns), y_train
            )
             payload = {
                "timestamp": timestamp,
                "features": features
            }
            st.session_state.selected_features = payload
            #save_selected_features("logistic_ffs", selected_features)
    
            X_train = pd.DataFrame(X_train_full, columns=X_df.columns)[selected_features]
            X_test = pd.DataFrame(X_test_full, columns=X_df.columns)[selected_features]
    
            models = ap.train_models(X_train, y_train)
            model_scores = ap.evaluate_models(models, X_test, y_test)
    
            scores_df = pd.DataFrame(model_scores).T.reset_index().rename(columns={"index": "Model"})
            scores_melted = scores_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
            st.session_state.scores_df = scores_df
            fig = px.bar(
                scores_melted,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="📊 Model Performance Across Metrics"
            )
            st.session_state.fig = fig
            ap.select_best_model(model_scores, metric="AUC")   

        
    
        # ✅ Mark pipeline as completed
        st.session_state.pipeline_ran = True
        st.session_state.run_pipeline = False
        #st.experimental_rerun()

    if st.session_state.pipeline_ran:        
        st.subheader("📋 Model Metrics")
        st.dataframe(st.session_state.scores_df)
        st.plotly_chart(st.session_state.fig, use_container_width=True)
        with st.expander("📦 View saved features"):
            st.caption("✨Features used")
            st.json(st.session_state.selected_features)
        st.success("✅ Pipeline completed!")
                
    
        # Save to GitHub
        if st.button("💾 Save to GitHub"):
            st.write("start saving to github")
            save_selected_features("logistic_ffs", st.session_state.selected_features)
            saveToGit("logistic_ffs")
            
            # Confirmation message
            st.success("✅ Features saved to GitHub successfully!")
            st.toast("📁 logistic_ffs.json uploaded", icon="📤")


        

            
    
        


