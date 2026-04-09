import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

class Wav2VecFeatureExtractor:
    """
    Wrapper around Facebook Wav2Vec2 to extract embeddings from audio samples.
    """

    def __init__(self, model_name: str ="facebook/wav2vec2-base", pooling: str ="stats"):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()  # Inference mode

        # Select pooling method
        if pooling.lower() == "stats":
            self.pooling = self.statistics_pooling
        else:
            raise ValueError(f"Invalid pooling method: {pooling}")

    def extract_embedding(self, path: str) -> torch.Tensor:
        """
        Extracts a fixed size embedding from a single audio sample.

        Loads a waveform from the given filepath, ensures that it is sampled
        at 16 kHz, and passes it through a pretrained wav2vec model, obtaining
        hidden state representations. The middle four layers are then selected
        and pooled to produce the final embedding.

        Args:
            path (str): Path to the input audio file.

        Returns:
            torch.Tensor: Extracted embedding.

        Notes:
            - The pooling type is defined by 'self.pooling'
        """
        # Load wav
        y, sr = torchaudio.load(path)

        # TIMIT sr should be 16kHz, but resample if not
        if sr != 16000:
            y = torchaudio.transforms.Resample(sr, 16000)(y)
        y = y.squeeze(0)  # Remove channel dim if mono, model expects 1D waveform

        # Get wav2vec embedding
        inputs = self.feature_extractor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states

        # Get middle 4 layers
        layers = hidden_states[6:10]

        # Apply chosen pooling method
        embedding = self.pooling(layers)

        return embedding

    def build_embeddings(self, df) -> np.ndarray:
        """
        Builds a feature matrix of embeddings from the dataset dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing paths to audio samples.

        Returns:
            np.ndarray: Array of shape (n_samples, n_features) where each
            row corresponds to the embedding of a sample.
        """
        embeddings = []

        for path in df["filepath"]:
            emb = self.extract_embedding(path)
            embeddings.append(emb)

        X = np.vstack(embeddings)

        return X

    def statistics_pooling(self, layers) -> torch.Tensor:
        """
        Statistics / mean + std pooling over layers
        """
        layer_embeddings = []

        for layer in layers:

            mean = layer.mean(dim=1) # (B, 768)
            std = layer.std(dim=1) # (B, 768)
            stats = torch.cat([mean, std], dim=1) # (B, 1536)

            layer_embeddings.append(stats)

        # Concatenate across layers
        embedding = torch.cat(layer_embeddings, dim=1) # (B, 6144)

        return embedding
