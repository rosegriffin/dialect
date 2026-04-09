import numpy as np
from sklearn.metrics import classification_report

def report_classification(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Prints classification report for model predictions.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
    """
    print(classification_report(y_true, y_pred))
