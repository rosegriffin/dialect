from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_logistic_regression(X_train: np.ndarray,
                              y_train: np.ndarray,
                              param_grid={"C": [0.001, 0.01, 0.1, 1, 10, 100]},
                              cv= 5):
    """
    Train Logistic Regression model with optional hyperparameter search.

    Args:
        X_train (np.ndarray): Training feature array
        y_train (np.ndarray): Training labels
        param_grid (dict, optional): Grid of parameters for GridSearchCV.
        cv (int): Number of cross validation folds. Defaults to 5.

    Returns:
        best_model: Trained LogisticRegression estimator with best parameters
        grid: GridSearchCV object
    """
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f'Best parameters: {grid.best_params_}\n')

    return best_model, grid
