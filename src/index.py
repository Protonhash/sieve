"""Index module: builds and queries FAISS approximate nearest neighbor indices."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import faiss
import numpy as np

logger = logging.getLogger(__name__)

IndexType = Literal["flat", "ivf", "hnsw"]


class IndexBuilder:
    """Wraps FAISS index construction with optional GPU acceleration."""

    def __init__(
        self,
        dim: int,
        index_type: IndexType = "ivf",
        use_gpu: bool = False,
        nlist: int = 4096,
        M: int = 64,
        nprobe: int = 32,
    ) -> None:
        self.dim = dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.nlist = nlist
        self.M = M
        self.nprobe = nprobe
        self._index: faiss.Index | None = None
        self._gpu_resources: faiss.GpuResources | None = None

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("Index not built yet. Call build() first.")
        return self._index

    def build(self, vectors: np.ndarray) -> None:
        """Build a FAISS index from a (n, dim) float32 array."""
        n, d = vectors.shape
        assert d == self.dim, f"Vector dim mismatch: {d} vs {self.dim}"
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        logger.info("Building %s index with %d vectors (dim=%d)", self.index_type, n, d)

        if self.index_type == "flat":
            cpu_index = faiss.IndexFlatIP(d)  # Inner product (cosine on normed vectors)

        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            cpu_index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.M, 8)
            logger.info("Training IVF index (nlist=%d, M=%d)...", self.nlist, self.M)
            cpu_index.train(vectors)

        elif self.index_type == "hnsw":
            cpu_index = faiss.IndexHNSWFlat(d, 32)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        cpu_index.add(vectors)

        if self.use_gpu:
            logger.info("Moving index to GPU...")
            self._gpu_resources = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(
                self._gpu_resources, 0, cpu_index
            )
        else:
            self._index = cpu_index

        self._index.nprobe = self.nprobe
        logger.info("Index ready: %d vectors", self._index.ntotal)

    def search(self, queries: np.ndarray, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Search the index. Returns (distances, indices)."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        return self.index.search(queries, k)

    def save(self, path: str | Path) -> None:
        """Serialize index to disk."""
        path = Path(path)
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(path))
        else:
            faiss.write_index(self.index, str(path))
        logger.info("Saved index to %s", path)

    def load(self, path: str | Path, use_gpu: bool | None = None) -> None:
        """Load a serialized FAISS index."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu
        cpu_index = faiss.read_index(str(path))
        if use_gpu:
            self._gpu_resources = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
        else:
            self._index = cpu_index
        self._index.nprobe = self.nprobe
        logger.info("Loaded index from %s (%d vectors)", path, self._index.ntotal)
