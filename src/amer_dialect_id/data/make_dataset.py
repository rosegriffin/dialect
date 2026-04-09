import os
import glob
import pandas as pd
from pathlib import Path

from amer_dialect_id.config import DATA_PROCESSED_ROOT, VALID_LEVELS

def parse_path(path: str) -> dict:
    """
    Parses a filepath to extract sample attributes.
    TODO only handles utterances, add support for words and phonemes

    Args:
        path (str): Full path to a .WAV file. Assumes the path format:
                    .../<split>/<dialect>/<speaker_id>/<utterance>.WAV

    Returns:
        dict: Dictionary containing the following keys:
            - 'split': data split (i.e. train/test)
            - 'dialect': dialect region
            - 'speaker_id': speaker identifier
            - 'utterance': utterance identifier (i.e. SA1, SA2)
            - 'filepath': path to the sample
    """
    path = Path(path)
    relative_path = path.relative_to(DATA_PROCESSED_ROOT)
    parts = relative_path.parts

    level = parts[0]
    data_split = parts[1]
    dialect = parts[2]
    speaker_id = parts[3]
    utt = parts[4].replace(".WAV", "")

    attributes = {"split": data_split,
                 "dialect": dialect,
                 "speaker_id": speaker_id,
                 "utterance": utt,
                 "filepath": path}

    return attributes

def make_dataset(level: str, utterances: list = ["SA1", "SA2"]) -> pd.DataFrame:
    """
    Creates a dataset dataframe for a specified processing level.

    Args:
        level (str): Type of processed data to load.
                     Options: "utterances", "words", "phonemes".
        utterances (list): List of utterance ids to include in dataframe

    Returns:
        pd.DataFrame: DataFrame containing filepaths and attributes:
                        - 'split': data split (i.e. train/test)
                        - 'dialect': dialect region
                        - 'speaker_id': speaker id
                        - 'utterance': utterance id (i.e. SA1, SA2)
                        - 'filepath': path to the sample

    Raises:
        FileNotFoundError: If the 'processed' folder does not exist.
        ValueError: If 'level' is not a valid option.
    """
    if level not in VALID_LEVELS:
        raise ValueError(f'Invalid level {level}. Valid options are: {VALID_LEVELS}')

    folder_path = DATA_PROCESSED_ROOT / level
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f'Processed data folder not found: {folder_path}')

    paths = glob.glob(f'{folder_path}/*/*/*/*.WAV')

    rows = [parse_path(path) for path in paths]

    df = pd.DataFrame(rows)

    if utterances is not None:
        df = df[df["utterance"].isin(utterances)]
    df = df[df["dialect"] != "DR8"] # Drop army bat

    return df
