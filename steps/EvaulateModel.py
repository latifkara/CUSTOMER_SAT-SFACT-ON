import pandas as pd
from zenml import step
import logging

class EvaulateModel:
    """
    Evaulate a model
    """
    def __init__(self, model: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def evaluate_model(self):
        logging.info(f"Evaluating model...")
        return self.model.score(self.x_test, self.y_test)

@step
def evaluate_model(model: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    """
    Evaulate a model
    """
    evaluate_model = EvaulateModel(model, x_test, y_test)
    return evaluate_model.evaluate_model()
        