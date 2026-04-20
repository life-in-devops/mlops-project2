import os
import joblib

from preprocess import (
    load_data,
    clean_data,
    feature_engineering,
    encode,
    split
)

from config import (
    DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PARAMS
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    print("🚀 Starting training pipeline...")

    # 🔹 Load data
    df = load_data(DATA_PATH)
    print(f"✅ Data loaded. Shape: {df.shape}")

    # 🔹 Preprocessing pipeline
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode(df)

    print(f"✅ Data after preprocessing: {df.shape}")

    # 🔹 Split features/target
    X, y = split(df, "Churn")

    # 🔹 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 🔹 Model initialization
    model = RandomForestClassifier(
        **MODEL_PARAMS,
        random_state=RANDOM_STATE
    )

    # 🔹 Train
    model.fit(X_train, y_train)
    print("✅ Model training completed")

    # 🔹 Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"🎯 Accuracy: {acc:.4f}")

    # 🔹 Save model
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print(f"💾 Model saved at: {model_path}")


if __name__ == "__main__":
    main()