import os
import joblib
import mlflow
import mlflow.sklearn

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
    print("🚀 Starting MLflow training pipeline...")

    # 🔹 Set experiment
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():

        # 🔹 Load data
        df = load_data(DATA_PATH)

        # 🔹 Preprocessing
        df = clean_data(df)
        df = feature_engineering(df)
        df = encode(df)

        X, y = split(df, "Churn")

        # 🔹 Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        # 🔹 Model
        model = RandomForestClassifier(
            **MODEL_PARAMS,
            random_state=RANDOM_STATE
        )

        # 🔹 Train
        model.fit(X_train, y_train)

        # 🔹 Predict
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"🎯 Accuracy: {acc:.4f}")

        # 🔹 Log parameters
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)

        # 🔹 Log metrics
        mlflow.log_metric("accuracy", acc)

        # 🔹 Save model locally
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)

        # 🔹 Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        print("✅ Model logged to MLflow")


if __name__ == "__main__":
    main()