import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.dropna()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]

    return X, y