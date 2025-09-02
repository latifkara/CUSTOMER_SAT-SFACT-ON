from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from zenml import step
import logging

class HandleCategoricalMissingValues(BaseEstimator, TransformerMixin):
    """
    Handle categorical missing values
    Args:
        na_name: Whether to return information about columns with missing values
    Returns:
        pd.DataFrame with categorical missing values handled
    """
    def __init__(self, na_name=False):
        self.na_name = na_name
        self.fitted_columns_ = None

    def missing_values_table(self, dataframe):
        """
        Identify columns with missing values and those with >10% missing values
        """
        na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        if not na_cols:
            return [], []
        
        n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        n_miss_cols = ratio[ratio > 10.0].index.tolist()
        return na_cols, n_miss_cols

    def categorical_values(self, dataframe):
        """
        Fill missing values in categorical columns with mode
        """
        dataframe = dataframe.copy()
        _, n_miss_cols = self.missing_values_table(dataframe)
        
        for col in n_miss_cols:
            if col in dataframe.columns:
                # Check if column has any non-null values
                if not dataframe[col].isnull().all():
                    mode_values = dataframe[col].mode()
                    if len(mode_values) > 0:
                        most_frequent = mode_values[0]
                        dataframe[col].fillna(most_frequent, inplace=True)
                        logging.info(f"Filled missing values in {col} with {most_frequent}")
                    else:
                        # If no mode can be calculated, fill with 'Unknown'
                        dataframe[col].fillna('Unknown', inplace=True)
                        logging.info(f"Filled missing values in {col} with 'Unknown'")
                else:
                    # If all values are null, fill with 'Unknown'
                    dataframe[col].fillna('Unknown', inplace=True)
                    logging.info(f"All values in {col} were null, filled with 'Unknown'")
        
        return dataframe

    def fit(self, X, y=None):
        """
        Fit the transformer by storing column information
        """
        self.fitted_columns_ = X.columns.tolist() if hasattr(X, 'columns') else None
        return self

    def transform(self, X):
        """
        Transform the data by handling categorical missing values
        """
        X = X.copy()  # Work on a copy of the dataframe
        
        # Ensure we're working with a DataFrame
        if not isinstance(X, pd.DataFrame):
            if self.fitted_columns_ is not None:
                X = pd.DataFrame(X, columns=self.fitted_columns_)
            else:
                X = pd.DataFrame(X)
        
        # Apply the categorical missing values handling
        X = self.categorical_values(X)
        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step
        """
        return self.fit(X, y).transform(X)

@step
def handle_categorical_missing_values(df: pd.DataFrame, na_name: bool = False) -> pd.DataFrame:
    """
    Handle categorical missing values
    Args:
        df: A pandas DataFrame
        na_name: Whether to return information about columns with missing values
    Returns:
        pd.DataFrame with categorical missing values handled
    """
    try:
        handler = HandleCategoricalMissingValues(na_name)
        result = handler.fit_transform(df)
        logging.info(f"Successfully handled missing values. Shape: {result.shape}")
        return result
    except Exception as e:
        logging.error(f"Error handling categorical missing values: {str(e)}")
        raise