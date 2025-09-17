# --- Main App ---
import streamlit as st
from settings import METADATA_PATH, DATA_PATH, MODEL_SAVE_DIR, MODEL_PERFORMANCE_PATH
import pandas as pd
import json
import plotly.express as px
from datetime import datetime
import pytz
from feature_store.registry import save_selected_features, saveToGit
from sklearn.model_selection import GridSearchCV
import time
import pickle
import os

# --- Dashboard Rendering ---
def show_model_history(path=METADATA_PATH):    
    
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
    st.session_state.current_model_name = current_model['version']
    st.sidebar.write(f"**Current Model**")
    st.sidebar.write(f"Version: {current_model['version']}")
    st.sidebar.write("ROC AUC: " + f"**{current_model['roc_auc']:.2%}**")
    st.sidebar.write(f"Activation Date: {current_model['date']}")   
    
    # Load and format performance data
    with open(MODEL_PERFORMANCE_PATH, "r") as f:
        m_perfomance = json.load(f)
    df_perfomance = pd.DataFrame(m_perfomance)
    
    # Convert timestamp and extract date
    df_perfomance['timestamp'] = pd.to_datetime(df_perfomance['timestamp'])
    df_perfomance['date'] = df_perfomance['timestamp'].dt.strftime('%d/%m/%y')
    
    # Group by date and calculate mean AUC
    auc_history_df = df_perfomance.groupby('date', as_index=False)['auc'].mean()
    
    # Sort by actual datetime to ensure correct order
    auc_history_df['sort_key'] = pd.to_datetime(auc_history_df['date'], format='%d/%m/%y')
    auc_history_df = auc_history_df.sort_values('sort_key')
    
    # Optional: drop sort_key if not needed
    auc_history_df.drop(columns='sort_key', inplace=True)
    
    # Display chart
    show_chart = st.toggle("📈 Show Model AUC Performance Over Time", value=False)
    if show_chart:
        st.subheader("Model AUC Performance Over Time")
        st.line_chart(auc_history_df.set_index('date')['auc']) 

    # Add model threshold to be set by the user
    st.sidebar.header("Model Monitoring")
    auc_threshold = st.sidebar.slider("Minimum AUC Threshold", 0.5, 1.0, 0.85, step=0.01)
    st.session_state.auc_threshold = auc_threshold
    #if below threshold run the automated pipeline
    #if current_model_auc < auc_threshold:
        #st.session_state.run_pipeline = True
        #st.session_state.pipeline_ran = False  

    # Add pipeline options on the UI sidebar
    st.sidebar.header("🔧 Feature Selection Options")
    # Selection method
    selection_method = st.sidebar.radio(
        "Choose feature selection method:",
        ["Forward Feature Selection (FFS)", "SHAP Top Features"]
    )
    
    # Number of features
    num_features = st.sidebar.number_input(
        "Number of features to select (min 7):",
        min_value=7,
        max_value=50,
        value=10,
        step=1
    )
    
    # Store in session state
    st.session_state.selection_method = selection_method
    st.session_state.num_features = num_features

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
    run_now = st.sidebar.button("🔄 Run Pipeline")

    if run_now:
        st.session_state.run_pipeline = True
        st.session_state.pipeline_ran = False
  
   
    # --- Pipeline Execution ---       
        
    if "start_time" not in st.session_state:
        st.session_state.start_time = None

    if run_now and not st.session_state.pipeline_ran:
        st.session_state.start_time = time.time()
        st.session_state.run_pipeline = False  # Immediately reset to prevent rerun loop
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
        # We need the below features for EX summary Tab1
        must_have = [
            "Contract", "InternetService", "TechSupport",
            "PaymentMethod", "Months", "MonthlyCharges", "TotalCharges"
        ]

        status.markdown("🧠 <span style='color:#ff7f0e'>Selecting features...</span>", unsafe_allow_html=True)
        
        if st.session_state.selection_method == "Forward Feature Selection (FFS)":
            selected_features, ffs_scores = ap.forward_feature_selection(
                pd.DataFrame(X_train_full, columns=X_df.columns),
                y_train,
                max_features=st.session_state.num_features,
                force_include=must_have
            )
        else:
            selected_features = ap.select_shap_top_features(
                pd.DataFrame(X_train_full, columns=X_df.columns),
                y_train,
                num_features=st.session_state.num_features,
                force_include=must_have
            )
        
        # Timestamp and payload
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
        models, grid_search = ap.train_models(X_train, y_train, X_test, y_test, st.session_state.current_model_name)
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
        st.session_state.best_model_auc = model_scores[st.session_state.best_model]["AUC"]
        st.session_state.best_model_index = model_scores.index
        stage_times.append(("Model Evaluation", time.time() - t0))
        progress.progress(80)
    
        # Stage 5: Finalization
        t0 = time.time()
        status.markdown("✅ <span style='color:#9467bd'>Finalizing pipeline and saving results...</span>", unsafe_allow_html=True)
        st.session_state.pipeline_ran = True
        st.session_state.run_pipeline = False
        stage_times.append(("Finalization", time.time() - t0))
        progress.progress(100)       
        end_time = time.time()
        elapsed = end_time - st.session_state.start_time
        stage_times.append(("Total execution time", elapsed))    
        st.session_state.stage_times = stage_times        
        status.markdown("🎉 <span style='color:green'>Pipeline completed successfully!</span>", unsafe_allow_html=True)
    

    if st.session_state.pipeline_ran: 
        end_time = time.time()
        elapsed = end_time - st.session_state.start_time  
        if st.session_state.best_model_auc > st.session_state.auc_threshold:
            annotation = "🆕 (new model)" if st.session_state.best_model == st.session_state.new_model else ""
            st.success(
                f"✅ Pipeline completed, with best model: {st.session_state.best_model} {annotation} and AUC: {st.session_state.best_model_auc:.4f}"
            )
        else:
            st.error(f"⚠️ Pipeline completed, with best model: {st.session_state.best_model} and AUC: {st.session_state.best_model_auc:.4f}")
        with st.expander("📋 Model Metrics"):            
            st.dataframe(st.session_state.scores_df)
            st.plotly_chart(st.session_state.fig, use_container_width=True)
        with st.expander("📦 View saved features"):
            st.caption("✨Features used")
            st.json(st.session_state.selected_features)
        with st.expander("🔍 View Grid search HPO"):
            st.caption("🔧 HPO used")
            st.write(st.session_state.grid_search.best_params_)        
        with st.expander("⏱️ Pipeline Timing Summary"):
            if "stage_times" in st.session_state:
                summary_df = pd.DataFrame(st.session_state.stage_times, columns=["Stage", "Time (s)"])                
                st.dataframe(summary_df.style.format({"Time (s)": "{:.2f}"}))           
        
       
        # Save to GitHub
        if st.sidebar.button("🚀 Deploy new model"):
            st.sidebar.success("Start saving to GitHub...")     
            # Save model locally
                 
            best_model = st.session_state.scores_df.loc[
                st.session_state.scores_df["Model"] == st.session_state.best_model
            ].iloc[0]
        
            model_obj = st.session_state.grid_search.best_estimator_
            model_filename = f"{MODEL_SAVE_DIR}/{st.session_state.best_model}.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(model_obj, f)
                
            if "model_saved" not in st.session_state:
                st.session_state.model_saved = True
            
            if st.session_state.model_saved:
                st.toast(f"📦 Model saved: {model_filename}", icon="💾")
                if st.button("Dismiss Toast"):
                    st.session_state.model_saved = False
        
            # Update model registry            
            tz_sydney = pytz.timezone("Australia/Sydney")
            timestamp = datetime.now(tz_sydney).strftime("%Y-%m-%d %H:%M:%S %Z")
        
            # Load existing registry
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r") as f:
                    registry = json.load(f)
            else:
                registry = []
        
            # Deactivate previous models
            for entry in registry:
                entry["active"] = False
        
            # Add new model entry
            new_entry = {
                "version": st.session_state.best_model,
                "date": timestamp,
                "accuracy": best_model["Accuracy"],
                "roc_auc": best_model["AUC"],
                "precision": best_model["Precision"],
                "recall": best_model["Recall"],
                "f1": best_model["F1"],
                "brier_score": best_model["Brier Score"],
                "features": st.session_state.selected_features["features"],
                "hyperparameters": st.session_state.grid_search.best_params_,
                "active": True,
                "notes": "Auto-deployed via Streamlit"
            }
            registry.append(new_entry)
        
            # Save updated registry
            with open(METADATA_PATH, "w") as f:
                json.dump(registry, f, indent=2)
        
            st.sidebar.success("✅ Model registry updated and activated!")
            st.toast("📘 Registry entry saved", icon="📚", duration=10)
            
            # Save selected features
            save_selected_features("logistic_ffs", st.session_state.selected_features)
            saveToGit("logistic_ffs", model_obj, model_filename)
            st.sidebar.success("✅ Features saved to GitHub successfully!")
            st.toast("📁 logistic_ffs.json uploaded", icon="📤", duration=10)





        

            
    
        


