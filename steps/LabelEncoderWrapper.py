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
    def __init__(self, X, cat_cols):
        self.encoders_ = {}
        self.columns_ = None
        self.X = X.copy()
        self.binary_cols = [col for col in cat_cols if self.X[col].nunique() == 2]
        self.X = X[self.binary_cols]


    def fit(self, y=None):

        for col in self.X.columns:
            le = LabelEncoder()
            le.fit(self.X[col].astype(str))
            self.encoders_[col] = le
        return self

    def transform(self):
        for col in self.X.columns:
            le = self.encoders_.get(col)
            if le:
                self.X[col] = le.transform(self.X[col].astype(str))  # Transform using stored encoder
        return self.X

@step
def label_encoder_wrapper(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    LabelEncoderWrapper
    Args:
        df: A pandas DataFrame
    Returns:
        pd.DataFrame with label encoded columns
    """
    label_encoder_wrapper = LabelEncoderWrapper(df, cat_cols)
    return label_encoder_wrapper.transform()