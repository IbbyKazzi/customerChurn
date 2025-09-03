# --- Main App ---
import streamlit as st
from settings import METADATA_PATH, DATA_PATH
import pandas as pd
import json
import plotly.express as px
from datetime import datetime
import pytz
from feature_store.registry import save_selected_features, saveToGit
from sklearn.model_selection import GridSearchCV
import time


# --- Dashboard Rendering ---
def show_model_history(path=METADATA_PATH):

    # Add model threshold to be set by the user
    st.sidebar.header("Model Monitoring")
    auc_threshold = st.sidebar.slider("Minimum AUC Threshold", 0.5, 1.0, 0.85, step=0.01)
    #if below threshold run the automated pipeline
    if current_model_auc < auc_threshold:
        st.session_state.run_pipeline = True
        st.session_state.pipeline_ran = False  

    # Add pipeline options on the UI sidebar
    st.sidebar.header("🔧 Feature Selection Options")
    # Selection method
    selection_method = st.sidebar.radio(
        "Choose feature selection method:",
        ["Forward Feature Selection (FFS)", "SHAP Top Features"]
    )
    
    # Number of features
    num_features = st.sidebar.number_input(
        "Number of features to select (min 5):",
        min_value=5,
        max_value=50,
        value=10,
        step=1
    )
    
    # Store in session state
    st.session_state.selection_method = selection_method
    st.session_state.num_features = num_features
    
    # Get current models
    with open(path, "r") as f:
        metadata = json.load(f)
    df = pd.DataFrame(metadata)

    # Display models stat 
    st.subheader("📋 Model Registry")
    st.dataframe(df[["version", "date", "accuracy", "roc_auc", "notes", "active"]])

    # Get current model AUC and display it
    current_model = df[df["active"] == True].iloc[0]
    current_model_auc = current_model['roc_auc']
    st.sidebar.write(f"**Current Model**")
    st.sidebar.write(f"Version: {current_model['version']}")
    st.sidebar.write("ROC AUC: " + f"**{current_model['roc_auc']:.0%}**")
    st.sidebar.write(f"Activation Date: {current_model['date']}")    
    
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
    start_time = time.time()
    if st.session_state.run_pipeline and not st.session_state.pipeline_ran:
        progress = st.progress(0)
        status = st.empty()
        stage_times = []
    
        import automated_pipeline as ap
    
        # Stage 1: Data Loading
        t0 = time.time()
        status.markdown("🔍 <span style='color:#1f77b4'>Loading and preprocessing data...</span>", unsafe_allow_html=True)
        X_df, y, (X_train_full, X_test_full, y_train, y_test) = ap.load_and_preprocess(DATA_PATH)
        stage_times.append(("Data Loading", time.time() - t0))
        progress.progress(20)
    
        # Stage 2: Feature Selection
        t0 = time.time()
        status.markdown("🧠 <span style='color:#ff7f0e'>Running forward feature selection...</span>", unsafe_allow_html=True)
        selected_features, ffs_scores = ap.forward_feature_selection(
            pd.DataFrame(X_train_full, columns=X_df.columns), y_train
        )
        tz_sydney = pytz.timezone("Australia/Sydney")
        timestamp = datetime.now(tz_sydney).strftime("%Y-%m-%d %H:%M:%S %Z")
        payload = {"timestamp": timestamp, "features": selected_features}
        st.session_state.selected_features = payload
        stage_times.append(("Feature Selection", time.time() - t0))
        progress.progress(40)
    
        # Stage 3: Model Training
        t0 = time.time()
        status.markdown("⚙️ <span style='color:#2ca02c'>Training models with Grid Search HPO...</span>", unsafe_allow_html=True)
        X_train = pd.DataFrame(X_train_full, columns=X_df.columns)[selected_features]
        X_test = pd.DataFrame(X_test_full, columns=X_df.columns)[selected_features]
        models, grid_search = ap.train_models(X_train, y_train, X_test, y_test)
        st.session_state.grid_search = grid_search
        stage_times.append(("Model Training", time.time() - t0))
        progress.progress(60)
    
        # Stage 4: Evaluation
        t0 = time.time()
        status.markdown("📊 <span style='color:#d62728'>Evaluating models...</span>", unsafe_allow_html=True)
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
        st.session_state.best_model = ap.select_best_model(model_scores, metric="AUC")
        stage_times.append(("Model Evaluation", time.time() - t0))
        progress.progress(80)
    
        # Stage 5: Finalization
        t0 = time.time()
        status.markdown("✅ <span style='color:#9467bd'>Finalizing pipeline and saving results...</span>", unsafe_allow_html=True)
        st.session_state.pipeline_ran = True
        st.session_state.run_pipeline = False
        stage_times.append(("Finalization", time.time() - t0))
        progress.progress(100)
        st.session_state.stage_times = stage_times
        status.markdown("🎉 <span style='color:green'>Pipeline completed successfully!</span>", unsafe_allow_html=True)
    

    if st.session_state.pipeline_ran:        
        st.subheader("📋 Model Metrics")
        st.dataframe(st.session_state.scores_df)
        st.plotly_chart(st.session_state.fig, use_container_width=True)
        with st.expander("📦 View saved features"):
            st.caption("✨Features used")
            st.json(st.session_state.selected_features)

        with st.expander("🔍 View Grid search HPO"):
            st.caption("🔧 HPO used")
            st.write(st.session_state.grid_search.best_params_)

        end_time = time.time()
        elapsed = end_time - start_time
        with st.expander("⏱️ Pipeline Timing Summary"):
            if "stage_times" in st.session_state:
                summary_df = pd.DataFrame(st.session_state.stage_times, columns=["Stage", "Time (s)"])                
                st.dataframe(summary_df.style.format({"Time (s)": "{:.2f}"}))

        st.success(f"✅ Pipeline completed in {elapsed:.2f} seconds")     
        
        #st.write(st.session_state.best_model)
        # Save to GitHub
        if st.button("🚀 Depoly new model"):
            st.success("Start saving to github...")
            save_selected_features("logistic_ffs", st.session_state.selected_features)
            saveToGit("logistic_ffs")
            
            # Confirmation message
            st.success("✅ Features saved to GitHub successfully!")
            st.toast("📁 logistic_ffs.json uploaded", icon="📤")


        

            
    
        


