import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import all model functions
from model.logistic_regression import train_and_evaluate_logistic_regression
from model.decision_tree import train_and_evaluate_decision_tree
from model.knn import train_and_evaluate_knn
from model.naive_bayes import train_and_evaluate_naive_bayes
from model.random_forest import train_and_evaluate_random_forest
from model.xgboost import train_and_evaluate_xgboost


st.set_page_config(page_title="Bank Marketing ML Models", layout="wide")

st.title("📊 Bank Marketing Classification - Model Comparison")
st.write("Compare multiple machine learning models for predicting term deposit subscription.")

# -------------------------------
# Model Selection
# -------------------------------

model_option = st.selectbox(
    "Select a Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# -------------------------------
# Dataset Upload
# -------------------------------

uploaded_file = st.file_uploader("Upload Bank Marketing CSV file", type=["csv"])

if uploaded_file is not None:

    # Read the file ONCE
    df = pd.read_csv(uploaded_file, sep=';')

    # Pass DataFrame to model functions
    if model_option == "Logistic Regression":
        model, metrics = train_and_evaluate_logistic_regression(df)

    elif model_option == "Decision Tree":
        model, metrics = train_and_evaluate_decision_tree(df)

    elif model_option == "KNN":
        model, metrics = train_and_evaluate_knn(df)

    elif model_option == "Naive Bayes":
        model, metrics = train_and_evaluate_naive_bayes(df)

    elif model_option == "Random Forest":
        model, metrics = train_and_evaluate_random_forest(df)

    elif model_option == "XGBoost":
        model, metrics = train_and_evaluate_xgboost(df)

    st.subheader("📈 Evaluation Metrics")

    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    st.dataframe(metrics_df)

    from model.preprocessing import load_and_preprocess_data, split_data

    X, y = load_and_preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    y_pred = model.predict(X_test)

    st.subheader("📉 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload the Bank Marketing dataset CSV file to proceed.")
