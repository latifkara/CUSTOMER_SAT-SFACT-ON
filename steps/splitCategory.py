# SplitCategory.py
import logging
from zenml import step
import pandas as pd
from typing import Tuple, List, Annotated


class SplitCategory:
    """
    Split the category column into categorical, numerical and cardinal columns
    Args:
        df: A pandas DataFrame
        cat_th: int = 10, car_th: int = 20
        cat_th: The threshold for categorical columns
        car_th: The threshold for cardinal columns
    Returns:
        Tuple of lists containing categorical, numerical and cardinal column names
    """
    def __init__(self, df: pd.DataFrame, cat_th: int = 10, car_th: int = 20):
        self.df = df
        self.cat_th = cat_th
        self.car_th = car_th

    def split_category(self) -> Tuple[List[str], List[str], List[str]]:
        # Categorical columns (object type)
        cat_cols = [col for col in self.df.columns if self.df[col].dtypes == "O"]
        
        # Numerical columns that should be treated as categorical (low unique values)
        num_but_cat = [col for col in self.df.columns if self.df[col].nunique() < self.cat_th and
                                                        self.df[col].dtypes != "O"]
        
        # Categorical columns that should be treated as cardinal (high unique values)
        cat_but_car = [col for col in self.df.columns if self.df[col].nunique() > self.car_th and
                                                        self.df[col].dtypes == "O"]

        # Final categorical columns
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # Numerical columns
        num_cols = [col for col in self.df.columns if self.df[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]
        num_cols = [col for col in num_cols if self.df[col].nunique() != len(self.df)]

        logging.info(f"Observations: {self.df.shape[0]}")
        logging.info(f"Variables: {self.df.shape[1]}")
        logging.info(f'cat_cols: {len(cat_cols)}')
        logging.info(f'num_cols: {len(num_cols)}')
        logging.info(f'cat_but_car: {len(cat_but_car)}')
        logging.info(f'num_but_cat: {len(num_but_cat)}')
        
        return cat_cols, num_cols, cat_but_car


# Method 1: Multiple outputs with Annotated (for newer ZenML versions)
@step
def split_category_multi(
    df: pd.DataFrame, 
    cat_th: int = 10, 
    car_th: int = 20
) -> Tuple[
    Annotated[List[str], "cat_cols"],
    Annotated[List[str], "num_cols"], 
    Annotated[List[str], "cat_but_car"]
]:
    """
    Split the category column into categorical, numerical and cardinal columns
    Args:
        df: A pandas DataFrame
        cat_th: The threshold for categorical columns
        car_th: The threshold for cardinal columns
    Returns:
        Tuple containing (cat_cols, num_cols, cat_but_car) as lists of column names
    """
    splitter = SplitCategory(df, cat_th, car_th)
    return splitter.split_category()


# Method 2: Single output with dictionary (more compatible)
@step
def split_category(df: pd.DataFrame, cat_th: int = 10, car_th: int = 20) -> dict:
    """
    Split the category column into categorical, numerical and cardinal columns
    Args:
        df: A pandas DataFrame
        cat_th: The threshold for categorical columns
        car_th: The threshold for cardinal columns
    Returns:
        Dictionary containing cat_cols, num_cols, and cat_but_car as lists
    """
    splitter = SplitCategory(df, cat_th, car_th)
    cat_cols, num_cols, cat_but_car = splitter.split_category()
    
    return {
        "cat_cols": cat_cols,
        "num_cols": num_cols, 
        "cat_but_car": cat_but_car
    }


# Method 3: Separate steps for data extraction
@step
def get_numerical_data(df: pd.DataFrame, column_info: dict) -> pd.DataFrame:
    """Extract numerical columns from dataframe"""
    num_cols = column_info["num_cols"]
    return df[num_cols] if num_cols else pd.DataFrame()


@step  
def get_categorical_data(df: pd.DataFrame, column_info: dict) -> pd.DataFrame:
    """Extract categorical columns from dataframe"""
    cat_cols = column_info["cat_cols"]
    return df[cat_cols] if cat_cols else pd.DataFrame()