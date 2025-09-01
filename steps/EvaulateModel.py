import pandas as pd
from zenml import step
import logging
from src.TrainTestSplit import TrainTestSplit

class EvaulateModel:
    """
    Evaulate a model
    """
    def __init__(self, model: pd.DataFrame, split: TrainTestSplit):
        self.model = model
        self.x_test = split.x_test
        self.y_test = split.y_test

    def evaluate_model(self):
        logging.info(f"Evaluating model...")
        return self.model.score(self.x_test, self.y_test)

@step
def evaluate_model(model: pd.DataFrame, split: TrainTestSplit) -> pd.DataFrame:
    """
    Evaulate a model
    """
    evaluate_model = EvaulateModel(model, split)
    return evaluate_model.evaluate_model()
        