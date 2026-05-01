import mlflow

TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "binance_rf_triple_barrier"


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
