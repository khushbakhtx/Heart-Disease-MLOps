import mlflow
from mlflow.tracking import MlflowClient
from fire import Fire

EXPERIMENT_NAME = 'first_experiment'
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

class HandleModel:
    def __init__(self, model_name: str, version: int = 1):
        self.model_name = model_name
        self.version = version
        self.client = MlflowClient()

    def register_model(self, run_id: str) -> None:
        model_uri = f"runs:/{run_id}/catboost_model_modified"
        mlflow.register_model(model_uri, self.model_name)

    def update_meta(self, description: str) -> None:
        self.client.update_model_version(
            name=self.model_name,
            version=self.version,
            description=description
        )

    def assign_alias(self, alias: str = 'staging') -> None:
        self.client.set_registered_model_alias(
            name=self.model_name, alias=alias, version=self.version
        )

    def tag_model(self, key: str = 'env', value: str = 'staging') -> None:
        self.client.set_model_version_tag(
            name=self.model_name, version=self.version, key=key, value=value
        )

if __name__ == "__main__":
    run_id = "608691c6f47a467ba25fb5657098869b"
    model_handler = HandleModel(model_name="catboost_model_v1.1")
    model_handler.register_model(run_id=run_id)
    model_handler.update_meta(description="classification model for heart disease")
    model_handler.assign_alias(alias="staging")
    model_handler.tag_model(key="env", value="staging")