from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from model.preprocessing import load_and_preprocess_data, split_data


def train_and_evaluate_decision_tree(df):
    # Load and preprocess data
    X, y = load_and_preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train Decision Tree model
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=None
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return model, metrics
