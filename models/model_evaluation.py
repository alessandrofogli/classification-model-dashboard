from sklearn.metrics import roc_auc_score

def evaluate_model(gs, X_train, X_test, y_train, y_test):
    """
    Evaluate the trained model on both training and test datasets.

    Parameters:
    - gs: A trained GridSearchCV object
    - X_test: Test feature dataset
    - y_test: Test target dataset

    Returns:
    - evaluation_results: A dictionary containing scores and the best parameters
    """
    train_score = roc_auc_score(y_train, gs.predict_proba(X_train)[:, 1])
    test_score = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])

    evaluation_results = {
        "Best Score of train set": train_score,
        "Test Score": test_score
    }

    return evaluation_results
