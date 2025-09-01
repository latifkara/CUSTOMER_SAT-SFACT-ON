import pandas as pd
from zenml import step
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class TrainModel:
    """
    Train a model
    Args:
        df: A pandas DataFrame
    Returns:
        A pandas DataFrame
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.importance_features = {}
        self.classifiers = [('DT', DecisionTreeClassifier()),
                ('RF', RandomForestClassifier()),
                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                ('LightGBM', LGBMClassifier(verbose=-1))]


    def split_data(self):
        X = self.df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        y = self.df['TARGET']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2024)
        return X_train, X_test, y_train, y_test
    
    def plot_importance(self, importance, features, num=10, save=False):
        feature_imp = pd.DataFrame({'Value': importance, 'Feature': features.columns})
        if save:
            plt.savefig('importances.png')

        return feature_imp['Feature'].head(num)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        for name, classifier in self.classifiers:
            logging.info(f"Model name: {name}")  
            model = classifier.fit(X_train, y_train)
            self.models[name] = model

        for name, model in self.models.items():
            features_cols = self.plot_importance(model.feature_importances_, X_train, 15)
            self.importance_features[name] = features_cols
            logging.info(f"Features columns: {features_cols}")

        for name, model in self.models.items():
            logging.info(f"Model {name} training....")
            self.models[name] = model.fit(X_train[self.importance_features[name]], y_train)
            
        logging.info(f"Models trained successfully")
        return self.models

@step
def train_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a model
    Args:
        df: A pandas DataFrame
    Returns:
        A pandas DataFrame
    """
    train_model = TrainModel(df)
    return train_model.train_model(), train_model.split_data()