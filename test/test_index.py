"""Tests for the FAISS index builder."""

import numpy as np
import pytest
from sieve.index import IndexBuilder


@pytest.fixture
def sample_vectors():
    rng = np.random.RandomState(42)
    vectors = rng.randn(1000, 384).astype(np.float32)
    # Normalize
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def test_build_flat(sample_vectors):
    builder = IndexBuilder(dim=384, index_type="flat", use_gpu=False)
    builder.build(sample_vectors)
    assert builder.index.ntotal == 1000


def test_build_ivf(sample_vectors):
    builder = IndexBuilder(dim=384, index_type="ivf", use_gpu=False, nlist=50)
    builder.build(sample_vectors)
    assert builder.index.ntotal == 1000


def test_search_flat(sample_vectors):
    builder = IndexBuilder(dim=384, index_type="flat", use_gpu=False)
    builder.build(sample_vectors)
    queries = sample_vectors[:10]
    distances, indices = builder.search(queries, k=5)
    assert distances.shape == (10, 5)
    assert indices.shape == (10, 5)
    # Self should be nearest neighbor (cosine sim ~ 1.0 on normalized vectors)
    assert np.allclose(distances[:, 0], 1.0, atol=1e-3)


def test_search_ivf(sample_vectors):
    builder = IndexBuilder(dim=384, index_type="ivf", use_gpu=False, nlist=50, nprobe=10)
    builder.build(sample_vectors)
    queries = sample_vectors[:10]
    distances, indices = builder.search(queries, k=5)
    assert distances.shape == (10, 5)
    assert indices.shape == (10, 5)


def test_save_load_flat(tmp_path, sample_vectors):
    builder = IndexBuilder(dim=384, index_type="flat", use_gpu=False)
    builder.build(sample_vectors)
    save_path = tmp_path / "test.index"
    builder.save(save_path)

    builder2 = IndexBuilder(dim=384, index_type="flat", use_gpu=False)
    builder2.load(save_path)
    assert builder2.index.ntotal == 1000
