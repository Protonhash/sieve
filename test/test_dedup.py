"""Integration tests for the dedup pipeline."""

import numpy as np
import pytest
from sieve.dedup import Deduplicator


@pytest.fixture
def deduplicator():
    return Deduplicator(
        embedder_name="paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold=0.85,
        index_type="flat",
        use_gpu=False,
    )


def test_deduplicate_no_duplicates(deduplicator, tmp_path):
    """Deduplication of clearly distinct texts should keep everything."""
    import json

    data = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Quantum mechanics describes the behavior of subatomic particles."},
        {"text": "The capital of France is Paris, a global center for art and fashion."},
        {"text": "Photosynthesis converts light energy into chemical energy in plants."},
    ]
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")

    manifest = deduplicator.deduplicate(str(input_file))
    assert manifest.total_passages == 4
    assert manifest.kept_count == 4
    assert manifest.duplicate_count == 0


def test_deduplicate_near_duplicates(deduplicator, tmp_path):
    """Near-duplicates should be detected."""
    import json

    data = [
        {"text": "The cat sat on the mat."},
        {"text": "A cat was sitting on the mat."},  # near-dup of 0
        {"text": "Machine learning is transforming every industry."},
        {"text": "The feline rested upon the rug."},  # near-dup of 0/1
    ]
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")

    manifest = deduplicator.deduplicate(str(input_file))
    # Should find at least one duplicate
    assert manifest.duplicate_count >= 1
    assert manifest.kept_count < 4


def test_deduplicate_multilingual(deduplicator, tmp_path):
    """Cross-lingual near-duplicates should be detected."""
    import json

    data = [
        {"text": "The weather is beautiful today."},
        {"text": "Le temps est magnifique aujourd'hui."},  # French translation
        {"text": "El clima es hermoso hoy."},              # Spanish translation
        {"text": "Blockchain technology enables decentralized finance."},
    ]
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")

    manifest = deduplicator.deduplicate(str(input_file))
    # Multilingual model should detect cross-lingual near-duplicates
    assert manifest.duplicate_count >= 1
