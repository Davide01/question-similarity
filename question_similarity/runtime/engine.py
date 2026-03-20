import mlflow
from mlflow.client import MlflowClient

from question_similarity.common.config import MLFLOW_TRACKING_URI


class Engine:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.model = mlflow.sentence_transformers.load_model("models:/test_model/1")

        # Get the run_id and metrics from the model version
        client = MlflowClient()
        model_version = client.get_model_version("test_model", 1)
        run_id = model_version.run_id

        # Get the run and extract metrics
        run = mlflow.get_run(run_id)
        try:
            self.final_similarity_accuracy = run.data.metrics[
                "final_test_cosine_accuracy_threshold"
            ]
        except KeyError:
            raise ValueError("Final similarity accuracy not found in model metrics.")

    def predict(self, question1: str, question2: str) -> bool:
        # Placeholder for prediction logic

        embeddings = self.model.encode([question1, question2])
        # Placeholder for similarity computation
        similarity = float(self.model.similarity(embeddings[0], embeddings[1]))

        return similarity > self.final_similarity_accuracy, similarity
