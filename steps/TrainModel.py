from tkinter import N
import pandas as pd
from zenml import step
import logging
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from src.TrainTestSplit import TrainTestSplit
from typing import Dict


class TrainModel:
    """
    Train a model
    Args:
        df: A pandas DataFrame
    Returns:
        A pandas DataFrame
    """
    def __init__(self, split: TrainTestSplit):
        self.split = split
        self.models = Dict()
        self.importance_features = {}
        self.classifiers = [('DT', DecisionTreeClassifier()),
                ('RF', RandomForestClassifier()),
                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                ('LightGBM', LGBMClassifier(verbose=-1))]


    def plot_importance(self, importance, features, num=10, save=False):
        feature_imp = pd.DataFrame({'Value': importance, 'Feature': features.columns})
        if save:
            plt.savefig('importances.png')

        return feature_imp['Feature'].head(num)

    def train_model(self):
        
        for name, classifier in self.classifiers:
            logging.info(f"Model name: {name}")  
            model = classifier.fit(self.split.X_train, self.split.y_train)
            self.models[name] = model

        for name, model in self.models.items():
            features_cols = self.plot_importance(model.feature_importances_, self.split.X_train, 15)
            self.importance_features[name] = features_cols
            logging.info(f"Features columns: {features_cols}")

        for name, model in self.models.items():
            logging.info(f"Model {name} training....")
            self.models[name] = model.fit(self.split.X_train[self.importance_features[name]], self.split.y_train)
            
        logging.info(f"Models trained successfully")
        return self.models

@step
def train_model(split: TrainTestSplit) -> Dict:
    """
    Train a model
    Args:
        df: A pandas DataFrame
    Returns:
        A pandas DataFrame
    """
    train_model = TrainModel(split)
    return train_model.train_model()