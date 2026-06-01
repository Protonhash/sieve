"""Sieve: Semantic deduplication for multilingual training corpora."""

__version__ = "0.2.1"
__author__ = "Protonhash"

from sieve.dedup import Deduplicator
from sieve.embed import Embedder
from sieve.index import IndexBuilder
from sieve.utils import load_config

__all__ = ["Deduplicator", "Embedder", "IndexBuilder", "load_config"]
