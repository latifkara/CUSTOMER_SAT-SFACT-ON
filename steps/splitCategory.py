import logging
from zenml import step
import pandas as pd


class SplitCategory:
    """
    Split the category column into categorical, numerical and cardinal columns
    Args:
        df: A pandas self.df, cat_th: int = 10, car_th: int = 20
        cat_th: The threshold for categorical columns
        car_th: The threshold for cardinal columns
    Returns:
        pd.DataFrame with categorical, numerical and cardinal columns
    """
    def __init__(self, df: pd.DataFrame, cat_th: int = 10, car_th: int = 20):
        self.df = df
        self.cat_th = cat_th
        self.car_th = car_th

    def split_category(self):
        cat_cols = [col for col in self.df.columns if self.df[col].dtypes == "O"]
        num_but_cat = [col for col in self.df.columns if self.df[col].nunique() < self.cat_th and
                                                        self.df[col].dtypes != "O"]
        cat_but_car = [col for col in self.df.columns if self.df[col].nunique() > self.car_th and
                                                        self.df[col].dtypes == "O"]

        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        #Number Columns
        num_cols = [col for col in self.df.columns if self.df[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]
        num_cols = [col for col in num_cols if self.df[col].nunique() != len(self.df)]

        logging.info(f"Observations: {self.df.shape[0]}")
        logging.info(f"Variables: {self.df.shape[1]}")
        logging.info(f'cat_cols: {len(cat_cols)}')
        logging.info(f'num_cols: {len(num_cols)}')
        logging.info(f'cat_but_car: {len(cat_but_car)}')
        logging.info(f'num_but_cat: {len(num_but_cat)}')
        
        dataframe = pd.DataFrame(data=pd.concat([cat_cols, num_cols, cat_but_car], axis=0), columns=["cat_cols", "num_cols", "cat_but_car"])
        return dataframe

@step
def split_category(df: pd.DataFrame, cat_th: int = 10, car_th: int = 20) -> pd.DataFrame:
    """
    Split the category column into categorical, numerical and cardinal columns
    Args:
        df: A pandas DataFrame
        cat_th: The threshold for categorical columns
        car_th: The threshold for cardinal columns
    Returns:
        pd.DataFrame with categorical, numerical and cardinal columns
    """
    split_category = SplitCategory(df, cat_th, car_th)
    return split_category.split_category()