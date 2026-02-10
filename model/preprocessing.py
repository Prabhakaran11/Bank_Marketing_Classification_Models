import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.read_csv("data/bank.csv", sep=';').head()

def load_and_preprocess_data(csv_path):
    """
    Loads the Bank Marketing dataset and performs preprocessing:
    - Encodes categorical variables
    - Scales numerical variables
    - Encodes target variable
    """

    # Load dataset (semicolon separated)
    df = pd.read_csv(csv_path, sep=';')

    # Encode target variable
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    # Label encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scale numerical variables
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into train and test sets using stratification
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )