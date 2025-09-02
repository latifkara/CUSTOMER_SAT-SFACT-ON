import pandas as pd
from zenml import step
import logging
from src.TrainTestSplitData import TrainTestSplitData
from sklearn.ensemble import VotingClassifier

class EvaulateModel:
    """
    Evaluate a model
    """
    def __init__(self, model: VotingClassifier, split: TrainTestSplitData):
        self.model = model
        self.X_test = split.X_test
        self.y_test = split.y_test

    def evaluate_model(self):
        logging.info("Evaluating model...")
        
        score = self.model.score(self.X_test, self.y_test)
        
        # Tek skor değerini DataFrame'e çevir
        df = pd.DataFrame({"Score": [score]})
        return df


@step
def evaluate_model(model: VotingClassifier, split: TrainTestSplitData) -> pd.DataFrame:
    """
    Evaulate a model
    """
    evaluate_model = EvaulateModel(model, split)
    return evaluate_model.evaluate_model()
        