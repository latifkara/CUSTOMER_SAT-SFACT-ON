import logging
from zenml import step
import pandas as pd

class OneHotEncoder:
    """
    One hot Encoder
    Args:
        df: A pandas DataFrame
    
    """
    def __init__(self, df: pd.DataFrame, cat_cols: list):
        self.df = df
        self.one_hot_cols = [col for col in cat_cols if 10 >= self.df[col].nunique() > 2]
        # self.df = self.df[one_hot_cols]

    def transform(self):
        
        one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
        self.df[self.one_hot_cols] = one_hot_encoder.fit_transform(self.df[self.one_hot_cols])
        return self.df

@step
def one_hot_encoder(df: pd.DataFrame, cat_cols: list):
    """
    One hot Encoder
    Args:
        df: A pandas DataFrame
    """
    one_hot_encoder = OneHotEncoder(df, cat_cols)
    return one_hot_encoder.transform()
    