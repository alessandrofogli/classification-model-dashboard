import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file into a DataFrame."""
    return pd.read_csv(filepath, index_col=0)

def preprocess_data(df, target_column, positive_class):
    """
    Perform initial preprocessing on the DataFrame.

    Parameters:
    - df: DataFrame to preprocess.
    - target_column: The name of the column to be converted into binary.
    - positive_class: The value in the target column that should be considered as '1' (positive class).

    Returns:
    - df: DataFrame with the target column converted to binary format.
    """
    df[target_column] = df[target_column].apply(lambda x: 1 if x == positive_class else 0)
    return df

