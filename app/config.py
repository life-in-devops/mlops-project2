import os

DATA_PATH = os.getenv("DATA_PATH")

if not DATA_PATH:
    raise ValueError("DATA_PATH environment variable is not set")

TARGET_COLUMN = "Churn"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5
}