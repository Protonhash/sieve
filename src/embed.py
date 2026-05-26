"""Embedding module: wraps sentence-transformers for batch inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Embedder:
    """Generates dense embeddings from text passages using a configurable
    SentenceTransformer model."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str | None = None,
        batch_size: int = 256,
        normalize: bool = True,
        use_fp16: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading embedder '%s' on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)

        if use_fp16 and device != "cpu":
            self.model.half()
            logger.info("Using fp16 inference")

        self.dim = self.model.get_sentence_embedding_dimension()
        self.device = device

    def embed(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        """Embed a sequence of texts, returning a (n, dim) float32 array."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_stream(
        self, batches: Iterator[Sequence[str]], total_batches: int | None = None
    ) -> Iterator[np.ndarray]:
        """Stream embeddings for batched input, yielding numpy arrays per batch."""
        for batch in tqdm(batches, total=total_batches, desc="Embedding"):
            yield self.embed(batch, show_progress=False)

    def embed_file(
        self,
        input_path: str | Path,
        text_column: str = "text",
        output_path: str | Path | None = None,
    ) -> np.ndarray:
        """Embed all texts from a JSONL or Parquet file."""
        from datasets import load_dataset

        input_path = Path(input_path)
        suffix = input_path.suffix

        if suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(input_path), split="train")
        elif suffix in (".parquet", ".pq"):
            dataset = load_dataset("parquet", data_files=str(input_path), split="train")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        texts = dataset[text_column]
        embeddings = self.embed(texts)

        if output_path is not None:
            np.save(output_path, embeddings)
            logger.info("Saved embeddings to %s", output_path)

        return embeddings
