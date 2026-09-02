import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistrar:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def register(self, output_path: str, run_name: str):
        with mlflow.start_run(run_name=run_name):

            logger.info(f"Registering model with parameters: {self.model.get_params()}")

            mlflow.log_params(
                {
                    "model_type": type(self.model).__name__,
                    "training_samples": len(self.X_train),
                }
            )
            mlflow.sklearn.log_model(self.model, artifact_path=output_path)
