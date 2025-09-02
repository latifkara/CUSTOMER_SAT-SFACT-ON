import logging
from zenml import step
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union, List, Optional
from src.TrainTestSplitData import TrainTestSplitData


class SplitData:
    """
    Split Data into train and test
    Args:
        df: A pandas DataFrame
        target_col: the columns of target variable
        test_size: size of test data split
        random_state: the number of seed for randomized split
        columns_to_drop: additional columns to drop besides target_col (can be None)
    Returns:
        TrainTestSplit class 
    """
    def __init__(self, df: pd.DataFrame, target_col: str, test_size: float = 0.33, 
                 random_state: int = 42, columns_to_drop: Optional[Union[str, List[str]]] = None) -> None:
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.columns_to_drop = columns_to_drop or []
        
        # Ensure columns_to_drop is a list
        if isinstance(self.columns_to_drop, str):
            self.columns_to_drop = [self.columns_to_drop]

    def split_data(self) -> TrainTestSplitData:
        """
        Perform the actual data splitting
        Returns:
            TrainTestSplit object containing train/test splits
        """
        try:
            # Prepare columns to drop
            cols_to_drop = [self.target_col]
            
            # Add additional columns to drop if they exist
            for col in self.columns_to_drop:
                if col in self.df.columns:
                    cols_to_drop.append(col)
                    logging.info(f"Dropping column: {col}")
                else:
                    logging.warning(f"Column {col} not found in dataframe, skipping...")
            
            # Check if SK_ID_CURR exists before dropping
            if 'SK_ID_CURR' in self.df.columns:
                cols_to_drop.append('SK_ID_CURR')
                logging.info("Dropping SK_ID_CURR column")
            else:
                logging.info("SK_ID_CURR column not found, skipping...")
            
            # Remove duplicates from cols_to_drop
            cols_to_drop = list(set(cols_to_drop))
            
            # Create features and target
            X = self.df.drop(cols_to_drop, axis=1)
            y = self.df[self.target_col]
            
            logging.info(f"Features shape: {X.shape}")
            logging.info(f"Target shape: {y.shape}")
            logging.info(f"Features columns: {list(X.columns)}")
            
            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y if len(y.unique()) > 1 else None  # Stratify if possible
            )
            
            logging.info(f"Train set shape: X_train {X_train.shape}, y_train {y_train.shape}")
            logging.info(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
            
            # Create and return TrainTestSplit object directly
            split = TrainTestSplitData(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )
            return split
            
        except Exception as e:
            logging.error(f"Error in split_data: {str(e)}")
            logging.error(f"DataFrame columns: {list(self.df.columns)}")
            logging.error(f"Target column: {self.target_col}")
            raise

@step
def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.33, 
               random_state: int = 42, columns_to_drop: Optional[Union[str, List[str]]] = None) -> TrainTestSplitData:
    """
    Split Data into train and test
    Args:
        df: A pandas DataFrame
        target_col: the columns of target variable
        test_size: size of test data split
        random_state: the number of seed for randomized split
        columns_to_drop: additional columns to drop besides target_col
    Returns:
        TrainTestSplit object 
    """
    try:
        logging.info(f"Starting data split with target column: {target_col}")
        logging.info(f"Test size: {test_size}, Random state: {random_state}")
        
        splitter = SplitData(df, target_col, test_size, random_state, columns_to_drop)
        result = splitter.split_data()
        
        logging.info("Data split completed successfully")
        return result
        
    except Exception as e:
        logging.error(f"Error in split_data step: {str(e)}")
        raise


