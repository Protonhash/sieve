"""Tests for the embedding module."""

import numpy as np
import pytest
from sieve.embed import Embedder


@pytest.fixture
def embedder():
    """Use a tiny model for tests to keep CI fast."""
    return Embedder(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        batch_size=8,
    )


def test_embed_empty(embedder):
    result = embedder.embed([])
    assert result.shape == (0, embedder.dim)


def test_embed_single(embedder):
    result = embedder.embed(["Hello world"])
    assert result.shape == (1, embedder.dim)
    assert result.dtype == np.float32


def test_embed_batch(embedder):
    texts = ["First sentence.", "Second sentence.", "Third one here."]
    result = embedder.embed(texts)
    assert result.shape == (3, embedder.dim)


def test_embed_normalized(embedder):
    result = embedder.embed(["test"])
    norm = np.linalg.norm(result[0])
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_embed_batch_larger_than_batch_size(embedder):
    texts = [f"Sentence number {i}" for i in range(20)]
    result = embedder.embed(texts)
    assert result.shape == (20, embedder.dim)
