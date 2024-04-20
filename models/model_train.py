import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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
    - gs: A trained GridSearchCV object
    """

    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0, n_jobs=1)

    clf_pipeline = Pipeline(steps=[
    ('col_trans', col_trans),
    ('drop_target', DropColumnTransformer(col_to_drop=-1)),  # Dropping the 'Risk' column
    ('model', clf)
    ])

    grid_params = {
        'model__max_depth': [3, 4, 5],
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': np.linspace(0.01, 0.2, 5)
    }

    gs = GridSearchCV(clf_pipeline, grid_params, cv=5, scoring='roc_auc', error_score='raise')
    gs.fit(X_train, y_train)
    
    return gs
