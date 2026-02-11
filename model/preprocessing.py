import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data(df):
    """
    Accepts a pandas DataFrame (already loaded)
    """

    df = df.copy()

    # Encode target
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    X = df.drop('y', axis=1)
    y = df['y']

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    from sklearn.preprocessing import LabelEncoder, StandardScaler

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
