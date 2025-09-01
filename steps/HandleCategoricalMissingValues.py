from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from zenml import step
import logging

class HandleCategoricalMissingValues(BaseEstimator, TransformerMixin):

    """
    Handle categorical missing values
    Args:
        na_name: Whether to return the columns with missing values
    Returns:
        pd.DataFrame with categorical missing values handled
    """
    def __init__(self, na_name=False):
        self.na_name = na_name

    def missing_values_table(self, dataframe):
        na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        n_miss_cols = ratio[ratio > 10.0].index
        return na_cols, n_miss_cols

    def categorical_values(self, dataframe):
        _, n_miss_cols = self.missing_values_table(dataframe)
        for col in n_miss_cols:
            most_frequent = dataframe[col].mode()[0]
            dataframe[col].fillna(most_frequent, inplace=True)
        return dataframe

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  # Work on a copy of the dataframe
        X = self.categorical_values(X)
        return X

@step
def handle_categorical_missing_values(df: pd.DataFrame, na_name: bool = False) -> pd.DataFrame:
    """
    Handle categorical missing values
    Args:
        df: A pandas DataFrame
        na_name: Whether to return the columns with missing values
    Returns:
        pd.DataFrame with categorical missing values handled
    """
    handle_categorical_missing_values = HandleCategoricalMissingValues(na_name)
    return handle_categorical_missing_values.transform(df)