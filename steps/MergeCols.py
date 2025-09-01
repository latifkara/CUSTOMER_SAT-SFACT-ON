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
    def __init__(self, df: pd.DataFrame, cat_cols: list, num_cols: list, cat_but_car: list):
        self.df = df
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_but_car = cat_but_car

    def merge_cols(self):
        self.df = pd.concat([self.cat_cols, self.num_cols, self.cat_but_car], axis=1)
        return self.df


@step
def merge_cols(df: pd.DataFrame, cat_cols: list, num_cols: list, cat_but_car: list) -> pd.DataFrame:
    """
    Merge cols
    """
    merge_cols = MergeCols(df, cat_cols, num_cols, cat_but_car)
    return merge_cols.merge_cols()