from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.impute import KNNImputer
from zenml import step
import logging
from typing import List
import numpy as np

class HandleNumericalMissingValues(BaseEstimator, TransformerMixin):
    """
    Handle numerical missing values
    Args:
        null_columns_name: Whether to return the columns with missing values
    Returns:
        pd.DataFrame with numerical missing values handled
    """
    def __init__(self, null_columns_name: bool = False):
        self.null_columns_name = null_columns_name
        self.dropped_cols_ = []
        self.imputer_ = None
        self.numerical_cols_ = []

    def missing_values_table(self, dataframe):
        """Calculate missing values statistics"""
        if dataframe.empty:
            return [], []
            
        na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        
        if not na_cols:  # No missing values
            return [], []
            
        ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        n_miss_cols = ratio[ratio > 30.0].index.tolist()
        
        return na_cols, n_miss_cols

    def fit(self, X, y=None):
        """Fit the imputer on the training data"""
        if X is None or X.empty:
            print("Warning: Empty DataFrame passed to fit")
            return self
            
        X = X.copy()
        print(f"Fit - Input shape: {X.shape}")
        print(f"Fit - Input columns: {list(X.columns)}")
        
        # Identify columns to drop (>10% missing values)
        _, n_miss_cols = self.missing_values_table(X)
        self.dropped_cols_ = n_miss_cols
        print(f"Fit - Columns to drop (>10% missing): {self.dropped_cols_}")
        
        # Remove columns with >10% missing values
        X_filtered = X.drop(self.dropped_cols_, axis=1, errors='ignore')
        print(f"Fit - Shape after dropping high-missing cols: {X_filtered.shape}\n")
        
        # Get numerical columns
        self.numerical_cols_ = X_filtered.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        print(f"Fit - Numerical columns found: {self.numerical_cols_}\n")
        
        # Only fit imputer if there are numerical columns with missing values
        if len(self.numerical_cols_) > 0:
            numerical_data = X_filtered[self.numerical_cols_]
            
            # Check if there are actually missing values in numerical columns
            missing_in_numerical = numerical_data.isnull().sum().sum()
            print(f"Fit - Missing values in numerical columns: {missing_in_numerical}\n")
            
            if missing_in_numerical > 0:
                self.imputer_ = KNNImputer(n_neighbors=min(5, len(X_filtered)))  # Ensure n_neighbors <= n_samples
                self.imputer_.fit(numerical_data)
                print("Fit - KNN Imputer fitted successfully")
            else:
                print("Fit - No missing values in numerical columns, no imputer needed")
        else:
            print("Fit - No numerical columns found")
        
        return self

    def transform(self, X):
        """Transform the data by imputing missing values"""
        if X is None or X.empty:
            print("Warning: Empty DataFrame passed to transform")
            return pd.DataFrame()
            
        X = X.copy()
        print(f"Transform - Input shape: {X.shape}")
        print(f"Transform - Input columns: {list(X.columns)}")
        print(f"Transform - Missing values before: {X.isnull().sum().sum()}")
        
        # Drop columns identified during fit
        X_result = X.drop(self.dropped_cols_, axis=1, errors='ignore')
        print(f"Transform - Shape after dropping cols: {X_result.shape}")
        
        # Apply imputation only to numerical columns if imputer exists
        if self.imputer_ is not None and len(self.numerical_cols_) > 0:
            # Check which numerical columns actually exist in current data
            existing_numerical_cols = [col for col in self.numerical_cols_ if col in X_result.columns]
            print(f"Transform - Existing numerical columns: {existing_numerical_cols}")
            
            if existing_numerical_cols:
                # Get numerical data
                numerical_data = X_result[existing_numerical_cols]
                
                # Check if there are missing values to impute
                missing_count = numerical_data.isnull().sum().sum()
                print(f"Transform - Missing values in numerical data: {missing_count}")
                
                if missing_count > 0:
                    # Apply imputation
                    imputed_values = self.imputer_.transform(numerical_data)
                    
                    # Create DataFrame with imputed values
                    imputed_df = pd.DataFrame(
                        imputed_values,
                        columns=existing_numerical_cols,
                        index=X_result.index
                    )
                    
                    # Update the numerical columns in the result dataframe
                    X_result[existing_numerical_cols] = imputed_df
                    print("Transform - Imputation completed")
                else:
                    print("Transform - No missing values to impute")
        else:
            print("Transform - No imputer available or no numerical columns")
        
        print(f"Transform - Final shape: {X_result.shape}")
        print(f"Transform - Missing values after: {X_result.isnull().sum().sum()}")
        print(f"Transform - Final columns: {list(X_result.columns)}")
        
        if X_result.empty:
            print("WARNING: Returning empty DataFrame!")
            print("Original columns:", list(X.columns))
            print("Dropped columns:", self.dropped_cols_)
            print("Numerical columns:", self.numerical_cols_)
        
        return X_result


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
    print(f"=== handle_numerical_missing_values STEP ===")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Input DataFrame columns: {list(df.columns)}")
    print(f"Input DataFrame dtypes:\n{df.dtypes}")
    print(f"Input missing values: {df.isnull().sum().sum()}")
    
    if df.empty:
        print("ERROR: Input DataFrame is empty!")
        return df
    
    try:
        handler = HandleNumericalMissingValues(null_columns_name)
        result = handler.fit(df).transform(df)
        
        print(f"Output DataFrame shape: {result.shape}")
        print(f"Output DataFrame columns: {list(result.columns)}")
        print(f"Remaining missing values: {result.isnull().sum().sum()}")
        
        if result.empty:
            print("CRITICAL ERROR: Output DataFrame is empty!")
            print("This suggests all columns were dropped or there was an error in processing")
            # Return the original DataFrame as fallback
            return df
        
        return result
        
    except Exception as e:
        logging.error(f"Error in handle_numerical_missing_values: {str(e)}")
        print(f"Error details: {str(e)}")
        print(f"DataFrame info:")
        print(df.info())
        import traceback
        traceback.print_exc()
        
        # Return original DataFrame as fallback
        print("Returning original DataFrame as fallback due to error")
        return df