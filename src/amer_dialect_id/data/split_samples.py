import re
import shutil
import argparse
import soundfile as sf
from pathlib import Path

from amer_dialect_id.config import PROJECT_ROOT, DATA_PROCESSED_ROOT

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "TIMIT"
OUTPUT_DIR = DATA_PROCESSED_ROOT
DEFAULT_TYPE = "word"
DEFAULT_FILENAMES = {"SA1", "SA2"}
SPLITS = {"TRAIN": "data/TRAIN",
          "TEST":  "data/TEST"}

def parse_args():
    parser = argparse.ArgumentParser(description="Splits TIMIT sentences by word or phone based on given sample number")
    parser.add_argument("-i", "--input-dir", default=DEFAULT_INPUT_DIR, type=str, help="Path to root of input directory")
    parser.add_argument("-t", "--type", default=DEFAULT_TYPE, choices=["word", "phone"], type=str, help="Segmentation type: 'wrd' or 'phn'")
    parser.add_argument("-s", "--sentences", default=DEFAULT_FILENAMES, nargs='+', type=str, help="Sentence filenames (ex. SA1, SX3, etc). Case sensitive")
    return parser.parse_args()

def normalize_label(label: str) -> str:
    """Makes label safe for filenames"""
    label = label.lower()
    label = re.sub(r"[^a-z0-9_-]", "", label)
    return label

def split_audio(sentences, input_dir, output_dir, speaker_id, split_type):
    """Splits audio files based on label sample number ranges defined in their corresponding transcription file."""
    extension = "WRD" if split_type == "word" else "PHN"

    #print(f'{input_dir}\n{speaker_dir}')
    
    for s in sentences:
        wav_path = input_dir / f'{s}.WAV'
        t_path = input_dir / f'{s}.{extension}'
        
        # Skip missing files
        if not (wav_path.exists() and t_path.exists()):
            print(f'{wav_path} or {t_path} does not exist')
            continue

        # Load wav
        audio, sr = sf.read(wav_path)
        assert sr == 16000, f"Expected 16000Hz, got {sr}Hz"

        # Open corresponding transcription file
        with open(t_path, "r") as f:
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, line in enumerate(f):
                # Get beginning and ending sample number for word/phone
                start, end, label = line.strip().split()
                start, end = int(start), int(end)

                # Cut audio
                segment = audio[start:end]

                # Determine filename
                label = normalize_label(label)
                out_name = f"{speaker_id}_{s}_{i:04}_{label}.wav"
                out_path = output_dir / out_name
                
                sf.write(out_path, segment, sr)
                print(f"Wrote {out_path} ({len(segment)} samples)")

if __name__ == "__main__":

    args = parse_args()
    split_type = "phone" if args.type == "phone" else "word"
    input_dir = Path(args.input_dir)
    output_dir = OUTPUT_DIR / (split_type + "s")
    sentences = set(args.sentences) # list --> set, remove duplicates

    # Check if passed in path exist
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory {args.input_dir} does not exist.')


    # Copy over utterances if the directory does not exist yet
    # Assumes that if the directory exists, its fully populated
    if not (DATA_PROCESSED_ROOT / "utterances").exists():
        shutil.copytree(input_dir / "data", DATA_PROCESSED_ROOT / "utterances")

    # Traverse TRAIN and TEST seperately
    for split_name, split_path in SPLITS.items():
        cur_path = input_dir / split_path

        # Traverse each dialect region directory
        for dialect in cur_path.iterdir():

            # Ignore files
            if not dialect.is_dir():
                continue

            # Traverse each speaker directory
            for speaker_id in dialect.iterdir():

                if not speaker_id.is_dir():
                    continue

                output_path = output_dir / split_name / dialect.name / speaker_id.name
                split_audio(sentences, speaker_id, output_path, speaker_id.name, split_type)
