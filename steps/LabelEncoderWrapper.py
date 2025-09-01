from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from zenml import step
import logging
from typing import Dict, Any, List, Union

class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    LabelEncoderWrapper for binary categorical columns
    Args:
        cat_cols: List of categorical columns to consider
    Returns:
        pd.DataFrame with label encoded binary columns
    """
    def __init__(self, cat_cols: List[str] = None):
        self.cat_cols = cat_cols or []
        self.encoders_ = {}
        self.binary_cols = []

    def fit(self, X, y=None):
        """
        Fit label encoders for binary categorical columns
        """
        self.binary_cols = [col for col in self.cat_cols if col in X.columns and X[col].nunique() == 2]
        
        for col in self.binary_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
            
        logging.info(f"Fitted LabelEncoder for {len(self.binary_cols)} binary columns: {self.binary_cols}")
        return self

    def transform(self, X):
        """
        Transform binary categorical columns using fitted encoders
        """
        X_transformed = X.copy()
        
        for col in self.binary_cols:
            if col in X_transformed.columns:
                le = self.encoders_.get(col)
                if le:
                    X_transformed[col] = le.transform(X_transformed[col].astype(str))
                    logging.info(f"Label encoded column: {col}")
                    
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step
        """
        return self.fit(X, y).transform(X)


# Method 1: Accept dictionary from split_category step
@step
def label_encoder_wrapper(df: pd.DataFrame, column_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply label encoding to binary categorical columns
    Args:
        df: A pandas DataFrame
        column_info: Dictionary containing column information from split_category
    Returns:
        pd.DataFrame with label encoded binary columns
    """
    cat_cols = column_info.get("cat_cols", [])
    
    encoder = LabelEncoderWrapper(cat_cols)
    encoded_df = encoder.fit_transform(df)
    
    logging.info("Label encoding completed")
    return encoded_df

