from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from zenml import step
import logging
from typing import Dict, Any, List

@step
def one_hot_encoder(df: pd.DataFrame, column_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply one-hot encoding to non-binary categorical columns
    Args:
        df: A pandas DataFrame
        column_info: Dictionary containing column information from split_category
    Returns:
        pd.DataFrame with one-hot encoded categorical columns
    """
    cat_cols = column_info.get("cat_cols", [])
    df_encoded = df.copy()
    
    # Find non-binary categorical columns (excluding those already label encoded)
    non_binary_cols = [col for col in cat_cols if col in df.columns and df[col].nunique() > 2]
    
    if not non_binary_cols:
        logging.info("No non-binary categorical columns found for one-hot encoding")
        return df_encoded
    
    # Apply one-hot encoding
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    
    # Get the categorical data
    cat_data = df_encoded[non_binary_cols]
    
    # Fit and transform
    encoded_array = encoder.fit_transform(cat_data)
    
    # Get feature names using get_feature_names_out()
    encoded_feature_names = encoder.get_feature_names_out(non_binary_cols)
    
    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(
        encoded_array, 
        columns=encoded_feature_names, 
        index=df_encoded.index
    )
    
    # Drop original categorical columns and add encoded columns
    df_final = df_encoded.drop(columns=non_binary_cols)
    df_final = pd.concat([df_final, encoded_df], axis=1)
    
    logging.info(f"One-hot encoding completed for {len(non_binary_cols)} columns")
    logging.info(f"Created {len(encoded_feature_names)} new encoded features")
    
    return df_final