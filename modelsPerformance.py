import pickle
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import streamlit as st


with open("model_top3.pkl", "rb") as f:
    model_t3 = pickle.load(f)

with open("model_all.pkl", "rb") as f:
    model_all = pickle.load(f)

# Get predicted probabilities
y_probs = model_t3.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Compute AUC score
auc_score = roc_auc_score(y_test, y_probs)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

st.subheader("📊 ROC Curve Performance")
st.pyplot(fig)
st.metric(label="ROC AUC Score", value=f"{auc_score:.2f}")
