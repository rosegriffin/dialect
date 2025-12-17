import os
import sys
import re
import argparse
import soundfile as sf

DEFAULT_INPUT_DIR = "../../data/raw/TIMIT"
DEFAULT_OUTPUT_DIR = "../../data/processed"
DEFAULT_TYPE = "word"
DEFAULT_FILENAMES = {"SA1", "SA2"}
SPLITS = {
    "TRAIN": "data/TRAIN",
    "TEST":  "data/TEST",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Splits TIMIT sentences by word or phone based on given sample number")
    parser.add_argument("-i", "--input-dir", default=DEFAULT_INPUT_DIR, type=str, help="Path to root of input directory")
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR, type=str, help="Path to root of output directory")
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
        wav_path = os.path.join(input_dir, f"{s}.WAV")
        t_path = os.path.join(input_dir, f"{s}.{extension}")
        
        # Assumes that if .wav or .wrd/.phn doesn't exist, the corresponding file also doesn't exist.
        if not (os.path.exists(wav_path) and os.path.exists(t_path)):
            print(f'{wav_path} and {t_path} do not exist')
            continue

        # Load wav
        audio, sr = sf.read(wav_path)
        assert sr == 16000, f"Expected 16000Hz, got {sr}Hz"

        # Open corresponding transcription file
        with open(t_path, "r") as f:
            os.makedirs(output_dir, exist_ok=True)
        
            for i, line in enumerate(f):
                # Get beginning and ending sample number for word/phone
                start, end, label = line.strip().split()
                start, end = int(start), int(end)

                # Cut audio
                segment = audio[start:end]

                # Determine filename and write to file
                label = normalize_label(label)
                out_name = f"{speaker_id}_{s}_{i:04}_{label}.wav"
                out_path = os.path.join(output_dir, out_name)
                
                sf.write(out_path, segment, sr)
                print(f"Wrote {out_path} ({len(segment)} samples)")

if __name__ == "__main__":

    args = parse_args()

    # Check if passed in paths exist
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        sys.exit(1)
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist.")
        sys.exit(1)

    split_type = "phone" if args.type == "phone" else "word"
    input_dir = args.input_dir
    output_dir = os.path.join(args.output_dir, split_type + "s")
    sentences = set(args.sentences) # list --> set, remove duplicates

    # Traverse TRAIN and TEST seperately
    for split_name, split_path in SPLITS.items():
        cur_path = os.path.join(input_dir, split_path)
        
        # Traverse each dialect region directory
        for dialect in os.listdir(cur_path):
            
            dialect_path = os.path.join(cur_path, dialect)
    
            if not os.path.isdir(dialect_path):
                print(f'{dialect_path} does not exist')
                continue

            # Traverse each speaker directory
            for speaker_id in os.listdir(dialect_path):
                speaker_dir = os.path.join(dialect_path, speaker_id)
                
                if not os.path.isdir(speaker_dir):
                    print(f'{speaker_dir} does not exist')
                    continue

                output_path = os.path.join(output_dir, split_name, dialect, speaker_id)
                split_audio(sentences, speaker_dir, output_path, speaker_id, split_type)
