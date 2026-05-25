#!/usr/bin/env bash
# Benchmark embedding throughput on CPU vs GPU
# Usage: bash scripts/benchmark_inference.sh [cpu|gpu]

set -euo pipefail

DEVICE="${1:-cpu}"
BATCH_SIZES=(32 64 128 256 512 1024)
NUM_PASSAGES=50000

echo "=== Sieve Embedding Benchmark ==="
echo "Device:       $DEVICE"
echo "Passages:     $NUM_PASSAGES"
echo ""

for bs in "${BATCH_SIZES[@]}"; do
    echo "--- batch_size=$bs ---"
    python -c "
import time
import numpy as np
from sieve.embed import Embedder

texts = [f'This is test passage number {i}. It contains some text for benchmarking purposes.' for i in range($NUM_PASSAGES)]

embedder = Embedder(
    model_name='paraphrase-multilingual-MiniLM-L12-v2',
    device='$DEVICE',
    batch_size=$bs,
)

start = time.perf_counter()
embeddings = embedder.embed(texts, show_progress=False)
elapsed = time.perf_counter() - start

print(f'  Time:        {elapsed:.2f}s')
print(f'  Throughput:  {len(texts)/elapsed:.0f} passages/sec')
print(f'  Embed shape: {embeddings.shape}')
print(f'  GPU memory:  not measured (run with nvidia-smi)')
"
    echo ""
done

echo "=== Done ==="
