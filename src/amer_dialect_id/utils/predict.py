import os
import argparse
import numpy as np

from amer_dialect_id.config import MODEL_DIR
from amer_dialect_id.features.wav2vec_features import Wav2VecFeatureExtractor
from amer_dialect_id.utils.persistence import save_object, load_object

def parse_args():
    parser = argparse.ArgumentParser(description="Predicts dialect for samples.")
    parser.add_argument("-m", "--model", type=str, choices=["wav2vec_lr", "mfcc_elm"], help="Model name", required=True)
    parser.add_argument("-s", "--samples", nargs='+', type=str, help="Sample paths (ex. SA1.WAV, SX3.WAV, etc). Case sensitive", required=True)

    return parser.parse_args()

def average_predictions(sample_probs):

    stacked_probs = np.vstack(sample_probs) # shape: (n_samples, n_classes)

    # Average probabilities across samples
    avg_probs = stacked_probs.mean(axis=0)  # shape: (n_classes,)

    # Determine final predicted class and confidence
    final_class = int(np.argmax(avg_probs))

    return final_class, avg_probs

def predict_sample(filepath: str, extractor, model, scaler) -> int:
    """
    Predicts the dialect of a single sample.

    Args:
        filepath (str): Path to the audio file
        extractor: Feature extractor object
        model: Trained classifier
        scaler: Fitted scaler

    Returns:
        prediction (int): Predicted class label
    """
    X = extractor.extract_embedding(filepath)

    if scaler is not None:
        X = scaler.transform(X)

    pred = model.predict_proba(X)

    return pred


def predict_batch(filepaths: list, extractor, model, scaler=None):

    sample_probs = [predict_sample(filepath, extractor, model, scaler) for filepath in filepaths]

    return average_predictions(sample_probs)

def ensure_paths_exist(sample_paths, model_path, scaler_path):
    """
    Ensures that all provided filepaths exist, raising an error if any are missing.

    Args:
        sample_paths (list): A list of filepaths to input samples.
        model_path (str): Path to the trained model file.
        scaler_path (str): Path to the model's corresponding scaler file.

    Raises:
        FileNotFoundError: If any of the provided paths do not exist.
    """
    for sample_path in sample_paths:
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f'File not found: {sample_path}')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'File not found: {model_path}')

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'File not found: {scaler_path}')

def label_to_name(label):
    mapping = {0 : "DR1 - New England",
               1 : "DR2 - Northern",
               2 : "DR3 - North Midland",
               3 : "DR4 - South Midland",
               4 : "DR5 - Southern",
               5 : "DR6 - New York City",
               6 : "DR7 - Western"}

    return mapping[label]

def format_class_probs(confs):
    lst = []

    for i, conf in enumerate(confs):
        lst.append((label_to_name(i), conf))

    lst.sort(key=lambda x: x[1], reverse=True)

    return lst

if __name__=="__main__":

    args = parse_args()
    model_name = args.model
    sample_paths = args.samples

    # Handle model choice
    if model_name == "wav2vec_lr":
        model_path = MODEL_DIR / "lr.pkl"
        scaler_path = MODEL_DIR / "wav2vec_scaler.pkl"
        extractor = Wav2VecFeatureExtractor()
    else:
        raise NotImplementedError(f'{model_name} is not implemented yet.')

    ensure_paths_exist(sample_paths, model_path, scaler_path)

    model = load_object(model_path)
    scaler = load_object(scaler_path)

    dialect, confs = predict_batch(sample_paths, extractor, model)

    print(f'Dialect: {label_to_name(dialect)}\n\nModel confidence by Region:')
    print('\n'.join(f'{conf[0]}: {conf[1]:.2f}' for conf in format_class_probs(confs)))
