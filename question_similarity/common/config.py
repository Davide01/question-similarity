import os

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = os.environ.get("MODEL_NAME", "test_model")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
