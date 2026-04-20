import pandas as pd

def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    # Convert TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop nulls
    df = df.dropna()

    # Drop ID column
    df = df.drop("customerID", axis=1)

    return df


def feature_engineering(df):
    df = df.copy()

    # Binary mapping
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Example feature
    df['AvgCharges'] = df['TotalCharges'] / (df['tenure'] + 1)

    return df


def encode(df):
    df = pd.get_dummies(df, drop_first=True)
    return df


def split(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y