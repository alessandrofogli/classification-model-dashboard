def evaluate_model(gs, X_test, y_test):
    """
    Evaluate the trained model on both training and test datasets.

    Parameters:
    - gs: A trained GridSearchCV object
    - X_test: Test feature dataset
    - y_test: Test target dataset

    Returns:
    - evaluation_results: A dictionary containing scores and the best parameters
    """
    train_score = gs.best_score_
    test_score = gs.score(X_test, y_test)
    best_params = gs.best_params_

    evaluation_results = {
        "Best Score of train set": train_score,
        "Best parameter set": best_params,
        "Test Score": test_score
    }

    return evaluation_results
