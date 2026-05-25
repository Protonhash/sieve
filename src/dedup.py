"""Core deduplication logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from datasets import Dataset, load_dataset

from sieve.embed import Embedder
from sieve.index import IndexBuilder, IndexType

logger = logging.getLogger(__name__)


@dataclass
class DedupManifest:
    """Output of a deduplication run."""
    total_passages: int
    duplicate_count: int
    kept_count: int
    duplicate_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    kept_indices: list[int] = field(default_factory=list)


class Deduplicator:
    """Main pipeline: chunk -> embed -> index -> search -> dedup."""

    def __init__(
        self,
        embedder_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.92,
        index_type: IndexType = "ivf",
        use_gpu: bool = False,
        batch_size: int = 256,
        k: int = 100,
        nlist: int = 4096,
        M: int = 64,
        nprobe: int = 32,
        device: str | None = None,
    ) -> None:
        self.embedder = Embedder(
            model_name=embedder_name,
            device=device,
            batch_size=batch_size,
        )
        self.similarity_threshold = similarity_threshold
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.k = k
        self.nlist = nlist
        self.M = M
        self.nprobe = nprobe

    def deduplicate(self, input_glob: str, text_column: str = "text") -> DedupManifest:
        """Run full deduplication on files matching input_glob."""
        # Load
        logger.info("Loading data from: %s", input_glob)
        dataset = load_dataset("json", data_files=input_glob, split="train")

        texts = dataset[text_column]
        n = len(texts)
        logger.info("Loaded %d passages", n)

        # Embed
        embeddings = self.embedder.embed(texts)

        # Index
        index_builder = IndexBuilder(
            dim=self.embedder.dim,
            index_type=self.index_type,
            use_gpu=self.use_gpu,
            nlist=self.nlist,
            M=self.M,
            nprobe=self.nprobe,
        )
        index_builder.build(embeddings)

        # Search
        logger.info("Running self-join KNN search (k=%d)...", self.k)
        distances, indices = index_builder.search(embeddings, k=self.k)

        # Filter duplicates
        manifest = self._filter_duplicates(distances, indices, n)
        logger.info(
            "Dedup complete: %d/%d removed (%.1f%%)",
            manifest.duplicate_count,
            manifest.total_passages,
            100 * manifest.duplicate_count / manifest.total_passages,
        )
        return manifest

    def _filter_duplicates(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        n: int,
    ) -> DedupManifest:
        """Identify duplicates above threshold using greedy longest-first."""
        kept = set(range(n))
        removed: set[int] = set()
        pairs: list[tuple[int, int, float]] = []

        # Sort passages by length (longest first) so we prefer keeping longer ones
        # In a real implementation we'd have passage lengths here; for now use index order
        candidates = list(range(n))

        for i in candidates:
            if i in removed:
                continue
            for j_idx in range(1, self.k):  # skip self (idx 0)
                j = int(indices[i, j_idx])
                if j == -1 or j == i:
                    continue
                if j in removed or j in kept:
                    continue
                sim = float(distances[i, j_idx])
                if sim >= self.similarity_threshold:
                    removed.add(j)
                    pairs.append((i, j, sim))

        kept_indices = sorted(kept - removed)
        return DedupManifest(
            total_passages=n,
            duplicate_count=len(removed),
            kept_count=len(kept_indices),
            duplicate_pairs=pairs,
            kept_indices=kept_indices,
        )

    def save_clean(
        self,
        manifest: DedupManifest,
        input_glob: str | None = None,
        dataset: Dataset | None = None,
        output_dir: str | Path = "output",
    ) -> None:
        """Save deduplicated dataset to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if dataset is None and input_glob is not None:
            dataset = load_dataset("json", data_files=input_glob, split="train")
        elif dataset is None:
            raise ValueError("Either input_glob or dataset must be provided")

        clean = dataset.select(manifest.kept_indices)
        clean.to_json(str(output_dir / "clean.jsonl"), orient="records", lines=True)
        logger.info("Saved %d clean passages to %s", len(clean), output_dir)
