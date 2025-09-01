from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from zenml import step
import logging

class PreserveColumnNames(BaseEstimator, TransformerMixin):
    """
    Preserve column names
    Args:
        columns_: The columns
    Returns:
        pd.DataFrame with preserved column names
    """
    def __init__(self):
        self.columns_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns_)

    def get_feature_names_out(self, input_features=None):
        return self.columns_

@step
def preserve_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve column names
    Args:
        df: A pandas DataFrame
    Returns:
        pd.DataFrame with preserved column names
    """
    preserve_column_names = PreserveColumnNames()
    return preserve_column_names.transform(df)