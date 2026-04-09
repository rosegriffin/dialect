import os
import logging
from transformers import logging as hf_logging

from amer_dialect_id.config import DATA_PROCESSED_ROOT, EMBEDDINGS_DIR, MODEL_DIR
from amer_dialect_id.features.wav2vec_features import Wav2VecFeatureExtractor
from amer_dialect_id.data.make_dataset import make_dataset
from amer_dialect_id.data.preprocess import get_labels, get_split, scale_features
from amer_dialect_id.utils.persistence import save_object, load_object
from amer_dialect_id.models.logistic_regression import train_logistic_regression
from amer_dialect_id.utils.metrics import report_classification

if __name__=="__main__":

    print("[INFO] Preparing dataframe.")
    df = make_dataset(level="utterances") # TODO only handles utterances,
                                          # add support for words and phonemes

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    train_emb_path = EMBEDDINGS_DIR / "train_embeddings.pkl"
    test_emb_path = EMBEDDINGS_DIR / "test_embeddings.pkl"

    # Check if embeddings were precomputed. If not, extract and save them
    if not os.path.exists(train_emb_path):
        print("[INFO] Wav2vec embeddings not found; initializing feature extractor and extracting embeddings.")
        extractor = Wav2VecFeatureExtractor()

        # Extract features
        X = extractor.build_embeddings(df)
        y = get_labels(df)

        # Split dataset
        X_train, y_train = get_split(X, y, df, split="train")
        X_test, y_test = get_split(X, y, df, split="test")

        # Save embeddings
        save_object(X_train, train_emb_path)
        save_object(X_test, test_emb_path)
        save_object(y_train, EMBEDDINGS_DIR / "y_train.pkl")
        save_object(y_test, EMBEDDINGS_DIR / "y_test.pkl")
    else:
        # Load precomputed embeddings
        X_train = load_object(train_emb_path)
        X_test = load_object(test_emb_path)
        y_train = load_object(EMBEDDINGS_DIR / "y_train.pkl")
        y_test = load_object(EMBEDDINGS_DIR / "y_test.pkl")

    # Preprocessing
    scaler_path = MODEL_DIR / "wav2vec_scaler.pkl"
    if not os.path.exists(scaler_path):
        print("[INFO] Scaler not found; initializing and fitting scaler.")
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        save_object(scaler, scaler_path)
    else:
        scaler = load_object(scaler_path)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # Training
    model_path = MODEL_DIR / "lr.pkl"
    if not os.path.exists(model_path):
        print("[INFO] Logistic Regression model not found; training model.")
        model, _ = train_logistic_regression(X_train_scaled, y_train)
        save_object(model, model_path)
    else:
        model = load_object(model_path)

    # Eval
    y_pred = model.predict(X_test_scaled)
    report_classification(y_test, y_pred)
