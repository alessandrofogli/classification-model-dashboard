from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
    
class CategoricalWOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, X, y=None):
        # No fitting needed for WOE calculation directly
        return self  

    def transform(self, X):
        X_transformed = X.copy()
        missing_value = -999  # Custom value to replace NaN values
        # Replace NaN with -999 before calculating WOE
        X = X.fillna(missing_value)
        
        # Assuming 'Risk' is not one of the columns to be transformed
        for column in X.columns:
            if column == self.target_col:
                continue

            # Calculate WOE values including the -999 values
            current_woe = calculate_woe(X[[column, self.target_col]], column, self.target_col)
            
            # Map WOE values back to the original DataFrame using 'map'
            woe_values = X[column].map(current_woe)
            
            # Replace the column in the transformed DataFrame
            X_transformed[column] = woe_values

        # Return transformed DataFrame excluding the target column
        return X_transformed
    
class NumericalWOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, n_bins=5):
        self.target_col = target_col
        self.n_bins = n_bins
        self.binner = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')

    def fit(self, X, y):
        # Fit the binner to all columns excluding the target
        self.binner.fit(X.drop(columns=[self.target_col]), y)
        return self

    def transform(self, X):
        # Exclude the target column for transformation
        features = X.drop(columns=[self.target_col])
        features_binned = pd.DataFrame(self.binner.transform(features), columns=features.columns, index=features.index)
        X_transformed = features_binned.copy()
        
        # Convert bins to string for WOE calculation
        for column in features.columns:
            X_transformed[column] = X_transformed[column].astype(str)
            # Combine the feature with target to calculate WOE
            feature_with_target = pd.concat([X_transformed[column], X[self.target_col]], axis=1)
            woe_values = calculate_woe(feature_with_target, column, self.target_col)
            X_transformed[column] = X_transformed[column].map(woe_values)
        
        return X_transformed

# Define the WOE calculation function for use in transformers
def calculate_woe(df, feature_col, target_col):
    if df[target_col].nunique() != 2:
        raise ValueError("Target column must be binary.")
    
    total_goods = df[target_col].sum()
    total_bads = df[target_col].count() - total_goods
    group_distribution = df.groupby(feature_col)[target_col].agg(['sum', 'count'])
    group_distribution['good'] = group_distribution['sum']
    group_distribution['bad'] = group_distribution['count'] - group_distribution['good']
    group_distribution['dist_good'] = group_distribution['good'] / total_goods
    group_distribution['dist_bad'] = group_distribution['bad'] / total_bads
    group_distribution['woe'] = np.log(group_distribution['dist_good'] / group_distribution['dist_bad'])
    group_distribution['woe'].replace({np.inf: 0, -np.inf: 0}, inplace=True)
    
    return group_distribution['woe']

class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col_to_drop):
        self.col_to_drop = col_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[self.col_to_drop] if isinstance(self.col_to_drop, str) else X.columns[self.col_to_drop])

def build_column_transformer(num_cols, cat_cols):
    """Build a column transformer for numeric and categorical columns."""

    num_pipeline = Pipeline([
        ('num_woe', NumericalWOETransformer(target_col='Risk', n_bins=5))
    ])

    cat_pipeline = Pipeline([
        ('cat_woe', CategoricalWOETransformer(target_col='Risk'))
    ])

    col_trans = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_cols),
        ('cat_pipeline', cat_pipeline, cat_cols)
    ], remainder='drop')

    return col_trans
