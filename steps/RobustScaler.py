import pandas as pd
from zenml import step
import logging
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler


class RobustScaler:
    """
    RobustScaler
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def transform(self):
        scaler = SklearnRobustScaler()
        scaled_data = scaler.fit_transform(self.df)
        self.df = pd.DataFrame(
            scaled_data, 
            columns=self.df.columns, 
            index=self.df.index
        )
        return self.df

@step
def robust_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """
    RobustScaler
    """
    robust_scaler = RobustScaler(df)
    return robust_scaler.transform()
