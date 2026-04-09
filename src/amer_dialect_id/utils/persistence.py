import os
import joblib


def save_object(obj, filepath: str):
    """
    Save an object (model, embeddings, etc) to a .pkl file.

    Args:
        obj: Object to save
        filepath (str): Path to save file (should end with .pkl)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)

    print(f"Saved {filepath}")


def load_object(filepath: str):
    """
    Load an object from a .pkl file.

    Args:
        filepath (str): Path to saved .pkl file

    Returns:
        Loaded object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    obj = joblib.load(filepath)
    print(f"Loaded {filepath}")

    return obj
