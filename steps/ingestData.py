import logging
from zenml import step
import pandas as pd

class IngestData:

    """
    Ingest data from a the path
    Args:
        data_path: The path to the data to ingest
    Returns:
        A pandas DataFrame
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingest data from a given path
    Args:
        data_path: The path to the data to ingest
    Returns:
        A pandas DataFrame
    """
    ingest_data = IngestData(data_path)
    return ingest_data.get_data()