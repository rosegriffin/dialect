import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

EMBEDDINGS_DIR = DATA_PROCESSED_ROOT / "embeddings"
MODEL_DIR = PROJECT_ROOT / "src" / "amer_dialect_id" / "models"

VALID_LEVELS = ["utterances", "words", "phonemes"]
