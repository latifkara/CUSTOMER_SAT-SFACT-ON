import logging
from zenml import step
import pandas as pd
from sklearn.model_selection import train_test_split
from src.TrainTestSplit import TrainTestSplit


class SplitData:
    """
    Split Data into train and test
    Args:
        df: A pandas DataFrame
        target_col: the columns of target variable
        test_size: size of test data split
        random_state: the number of seed for randomized split
    Returns:
        TrainTestSplit class 
    """
    def __init__(self, df, target_col: str, test_size: float = 0.33, random_state: int = 42) -> None:
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        X = self.df.drop([self.target_col, 'SK_ID_CURR'], axis=1)
        y = self.df[self.target_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return TrainTestSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


@step
def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.33, random_state: int = 42) -> pd.DataFrame:
    """
    Split Data into train and test
    Args:
        df: A pandas DataFrame
        target_col: the columns of target variable
        test_size: size of test data split
        random_state: the number of seed for randomized split
    Returns:
        TrainTestSplit class 
    """
    split_data = SplitData(df, target_col, test_size, random_state)
    return split_data.split_data()