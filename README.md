# Sieve

Find duplicate passages in multilingual text datasets. Not just exact copies — semantic duplicates too.

When you're building training corpora, you end up with the same information repeated across different sources, languages, and phrasings. MinHash misses these. Sieve catches them.

## How it works

1. Split documents into overlapping chunks
2. Embed each chunk with a multilingual model (default: `paraphrase-multilingual-MiniLM-L12-v2`)
3. Build a FAISS index (HNSW or IVF-PQ)
4. Self-join: find all pairs above your cosine similarity threshold
5. Output a deduplication manifest

The whole thing runs offline. No API calls, no cloud services.

```bash
sieve dedup ./corpus/ --threshold 0.85 --output ./clean/
```

## Language support

Works well: English, Chinese, Japanese, Korean, German, French, Spanish, Portuguese, Russian, Arabic.

Works OK (higher false-positive rate): Thai, Vietnamese, Hindi, Turkish, Indonesian.

The embedding model does most of the heavy lifting. If you find a better multilingual encoder, you can swap it in.

## Why not just MinHash?

MinHash is fast and works for exact or near-exact duplicates. But it completely misses:
- Translations of the same content
- Paraphrases
- Summaries of longer passages
- Same facts, different wording

Sieve is slower but catches these. Use both — MinHash for exact, Sieve for semantic.

## Performance

On a single A100, indexing 1M documents (~500M chunks):
- Embedding: ~4 hours
- Index build: ~20 minutes
- Query: ~15 minutes

On CPU (no GPU): multiply by ~10x for embedding, ~3x for query.

## Configuration

```yaml
# configs/default.yaml
model: paraphrase-multilingual-MiniLM-L12-v2
chunk_size: 256
overlap: 64
threshold: 0.85
index_type: hnsw
ef_construction: 200
```

## Design doc

See [DESIGN.md](DESIGN.md) for the full technical writeup — index structures, similarity metrics, tradeoffs.

## Dependencies

- Python 3.10+
- FAISS (CPU or GPU)
- sentence-transformers
- See `pyproject.toml`

```bash
pip install -e .
```

## Status

Early stage. Works end-to-end on moderate datasets (<10M docs). Scaling to 100M+ is the next big task. GPU index support is experimental.

MIT License.
