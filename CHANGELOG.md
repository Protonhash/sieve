# Changelog

## 0.2.0 — 2025-05-30
- GPU index support via FAISS GPU (experimental)
- Added HNSW index type (faster for <5M chunks)
- New: `sieve stats` command to analyze corpus before dedup
- Fixed: incorrect cosine similarity thresholding on IVF-PQ indexes
- Reduced memory usage by ~40% for large corpora (streaming embed)

## 0.1.1 — 2025-05-10
- Fixed: language detection false positives on code-heavy documents
- Added Indonesian and Thai to partial support list
- Improved chunk overlap handling at sentence boundaries

## 0.1.0 — 2025-04-28
- Initial release
- Multilingual semantic deduplication pipeline
- FAISS IVF-PQ index with self-join query
- CLI interface: `sieve dedup`, `sieve embed`, `sieve index`
- Tested on EN, ZH, JA, KO, DE, FR, ES, PT, RU, AR
