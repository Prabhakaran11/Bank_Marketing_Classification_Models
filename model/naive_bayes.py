from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import numpy as np

from model.preprocessing import load_and_preprocess_data, split_data


def train_and_evaluate_naive_bayes(df):
    # Load and preprocess data
    X, y = load_and_preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train Gaussian Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Adjust threshold to balance precision/recall
    threshold = (y_train == 1).sum() / len(y_train)
    
    # Make predictions with adjusted threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return model, metrics