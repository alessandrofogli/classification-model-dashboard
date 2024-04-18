import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file into a DataFrame."""
    return pd.read_csv(filepath, index_col=0)

def preprocess_data(df):
    """Perform initial preprocessing on the DataFrame."""
    df['Risk'] = np.where(df['Risk'] == 'good', 0, 1)
    return df
