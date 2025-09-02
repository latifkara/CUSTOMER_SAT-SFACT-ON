import pandas as pd
from zenml import step
import logging


class MergeCols:
    """
    Merge cols
    Args:
        df: A pandas DataFrame
        cat_cols: A list of categorical columns
        num_cols: A list of numerical columns
        cat_but_car: A list of cardinal columns
    Returns:
        A pandas DataFrame
    """
    def __init__(self, df_numerical: pd.DataFrame, df_categorical: pd.DataFrame, df_cardinal: pd.DataFrame):
        self.df_numerical = df_numerical
        self.df_categorical = df_categorical
        self.df_cardinal = df_cardinal

    def merge_cols(self):
        self.df = pd.concat([self.df_numerical, self.df_categorical, self.df_cardinal], axis=1)
        return self.df


@step
def merge_cols(df_numerical: pd.DataFrame, df_categorical: pd.DataFrame, df_cardinal: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cols
    """
    merge_cols = MergeCols(df_numerical, df_categorical, df_cardinal)
    return merge_cols.merge_cols()