from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from zenml import step
import logging

class HandleOutlier(BaseEstimator, TransformerMixin):
    """
    Handle outliers
    Args:
        lower_quarter: The lower quartile
        upper_quarter: The upper quartile
    Returns:
        pd.DataFrame with outliers handled
    """
    def __init__(self, lower_quarter=0.25, upper_quarter=0.75):
        self.lower_quarter = lower_quarter
        self.upper_quarter = upper_quarter

    def outlier_threshold(self, dataframe, col_name):
        quantile1 = dataframe[col_name].quantile(self.lower_quarter)
        quantile3 = dataframe[col_name].quantile(self.upper_quarter)
        interquantile_range = quantile3 - quantile1
        up_limit = quantile3 + 1.5 * interquantile_range
        low_limit = quantile1 - 1.5 * interquantile_range
        return up_limit, low_limit

    def fit(self, X, y=None):
        self.columns_ = X.columns.tolist()  # Record column names during fitting
        return self

    def transform(self, X):
        X = X.copy()
        for variable in X.columns:
            up_limit, low_limit = self.outlier_threshold(X, variable)
            X.loc[X[variable] < low_limit, variable] = low_limit
            X.loc[X[variable] > up_limit, variable] = up_limit
        return X

@step
def handle_outlier(df: pd.DataFrame, lower_quarter: float = 0.25, upper_quarter: float = 0.75) -> pd.DataFrame:
    """
    Handle outliers
    Args:
        df: A pandas DataFrame
        lower_quarter: The lower quartile
        upper_quarter: The upper quartile
    Returns:
        pd.DataFrame with outliers handled
    """
    handle_outlier = HandleOutlier(lower_quarter, upper_quarter)
    return handle_outlier.transform(df)