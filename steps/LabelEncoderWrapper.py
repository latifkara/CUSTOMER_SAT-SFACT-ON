from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from zenml import step
import logging

class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    LabelEncoderWrapper
    Args:
        encoders_: The encoders
        columns_: The columns
    Returns:
        pd.DataFrame with label encoded columns
    """
    def __init__(self):
        self.encoders_ = {}
        self.columns_ = None

    def fit(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            le = self.encoders_.get(col)
            if le:
                X[col] = le.transform(X[col].astype(str))  # Transform using stored encoder
        return X

@step
def label_encoder_wrapper(df: pd.DataFrame) -> pd.DataFrame:
    """
    LabelEncoderWrapper
    Args:
        df: A pandas DataFrame
    Returns:
        pd.DataFrame with label encoded columns
    """
    label_encoder_wrapper = LabelEncoderWrapper()
    return label_encoder_wrapper.transform(df)