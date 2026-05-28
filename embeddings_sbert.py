from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SBERTConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    show_progress_bar: bool = True
    device: str | None = None          # None = auto (GPU если есть, иначе CPU)
    normalize_embeddings: bool = True  # L2-нормализация для cosine = dot product


class TextEmbedder:

    def __init__(self, config: SBERTConfig | None = None):
        self.config = config or SBERTConfig()
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress_bar,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    cfg = SBERTConfig(model_name=model_name)
    return TextEmbedder(cfg).fit_transform(texts)


if __name__ == "__main__":
    docs = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",
        "Dogs love running in the park",
        "Stocks fell after the Fed announcement",
        "The Federal Reserve announcement caused stocks to fall",
    ]
    emb = embed_texts(docs)
    print(f"Shape: {emb.shape}")
    sims = emb @ emb.T
    print("Cosine similarity matrix:")
    print(np.round(sims, 3))
