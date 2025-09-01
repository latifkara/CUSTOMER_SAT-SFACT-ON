import pandas as pd
from zenml import step
import logging


class RobustScaler:
    """
    RobustScaler
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scaler = RobustScaler()

    def transform(self):
        self.df = self.scaler.fit_transform(self.df)
        return self.df

@step
def robust_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """
    RobustScaler
    """
    robust_scaler = RobustScaler(df)
    return robust_scaler.transform()
