# Roadmap

## Next up
- [ ] Streaming mode for 100M+ document corpora
- [ ] GPU index as default (CPU fallback for compatibility)
- [ ] Incremental indexing (add new docs without rebuilding full index)
- [ ] Benchmark suite with standard dedup datasets

## Considering
- [ ] Cross-lingual dedup (same content in different languages)
- [ ] Fuzzy URL dedup (documents from different URLs, same content)
- [ ] Integration with HuggingFace datasets library
- [ ] Distributed indexing across multiple machines

## Not planned
- API service / hosted version — this is a local tool
- Exact hash dedup (MinHash already does this well, use it alongside Sieve)
