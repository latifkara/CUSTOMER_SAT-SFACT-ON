import pandas as pd
from zenml import step
import logging
import mlflow
from src.TrainTestSplitData import TrainTestSplitData
from sklearn.ensemble import VotingClassifier
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

class EvaulateModel:
    """
    Evaluate a model
    """
    def __init__(self, model: VotingClassifier, train_out: dict, split: TrainTestSplitData):
        self.model = model
        self.processed_data = train_out.get('processed_data', {})
        self.X_test = self.processed_data["X_test"]
        self.y_test = split.y_test

    def evaluate_model(self):
        logging.info("Evaluating model...")

        score = self.model.score(self.X_test, self.y_test)
        mlflow.log_metric("score", score)
        
        # Tek skor değerini DataFrame'e çevir
        return float(score)


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: VotingClassifier, train_out: dict, split: TrainTestSplitData) -> float:
    """
    Evaulate a model
    """
    evaluate_model = EvaulateModel(model, train_out, split)
    return evaluate_model.evaluate_model()
        