import pandas as pd
from zenml import step
import logging
from sklearn.ensemble import VotingClassifier

class TrainVottingClassifier:
    """
    Train a votting classifier
    Args:
        models: A dictionary of models
        X_train: A pandas DataFrame
        y_train: A pandas DataFrame
    Returns:
        A voting classifier
    """
    def __init__(self, models: dict, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.models = models
        self.X_train = X_train
        self.y_train = y_train

    def train_votting_classifier(self):
        dt_model = self.models['DT']
        rf_model = self.models['RF']
        xgboost_model = self.models['XGBoost']
        light_model = self.models['LightGBM']
        # Create Voting Classifier
        voting_clf = VotingClassifier(estimators=[
            ('DT', dt_model),
            ('RF', rf_model),
            ('XGBoost', xgboost_model),
            ('LightGBM', light_model)
        ], voting='hard')

        voting_clf.fit(self.X_train, self.y_train)
        logging.info(f"Voting classifier trained successfully")
        return voting_clf
        
@step
def train_votting_classifier(models: dict, X_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
    """
    Train a votting classifier
    Args:
        models: A dictionary of models
        X_train: A pandas DataFrame
        y_train: A pandas DataFrame
    Returns:
        A voting classifier
    """
    train_votting_classifier = TrainVottingClassifier(models, X_train, y_train)
    return train_votting_classifier.train_votting_classifier()