import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from data_processing.preprocessing import build_column_transformer
from data_processing.preprocessing import DropColumnTransformer



def train_model(X_train, y_train, col_trans):
    """
    Train the machine learning model with GridSearchCV to find the best hyperparameters.

    Parameters:
    - X_train: Training feature dataset
    - y_train: Training target dataset
    - col_trans: Column transformer

    Returns:
    - clf_pipeline: fitted pipeline object
    """

    clf = XGBClassifier(eta=0.01, max_depth=3, eval_metric='logloss', random_state=0, n_jobs=1)

    clf_pipeline = Pipeline(steps=[
    ('col_trans', col_trans),
    ('drop_target', DropColumnTransformer(col_to_drop=-1)),  # Dropping the 'Risk' column
    ('model', clf)
    ])

    clf_pipeline.fit(X_train, y_train)
    
    return clf_pipeline
