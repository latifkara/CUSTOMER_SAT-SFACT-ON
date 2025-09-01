from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.impute import KNNImputer
from zenml import step
import logging

class HandleNumericalMissingValues(BaseEstimator, TransformerMixin):
    """
    Handle numerical missing values
    Args:
        na_name: Whether to return the columns with missing values
    Returns:
        pd.DataFrame with numerical missing values handled
    """
    def __init__(self, null_columns_name: bool = False):
        self.null_columns_name = null_columns_name
        self.dropped_cols_ = []  # To keep track of dropped columns

    def missing_values_table(self, dataframe):
        na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        # n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        n_miss_cols = ratio[ratio > 10.0].index
        if self.null_columns_name:
            return na_cols, n_miss_cols
        return na_cols, []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        _, n_miss_cols = self.missing_values_table(X)
        X = X.drop(n_miss_cols, axis=1, errors='ignore')  # Drop columns identified in fit
        imputer = KNNImputer(n_neighbors=5)
        na_num_cols = X.select_dtypes(include='float').columns
        X[na_num_cols] = pd.DataFrame(
            imputer.fit_transform(X[na_num_cols]),
            columns=X[na_num_cols].columns
        )
        return X


@step
def handle_numerical_missing_values(df: pd.DataFrame, null_columns_name: bool = False) -> pd.DataFrame:
    """
    Handle numerical missing values
    Args:
        df: A pandas DataFrame
        null_columns_name: Whether to return the columns with missing values
    Returns:
        pd.DataFrame with numerical missing values handled
    """
    handle_numerical_missing_values = HandleNumericalMissingValues(null_columns_name)
    return handle_numerical_missing_values.transform(df)