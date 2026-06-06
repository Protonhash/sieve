# Contributing

Sieve is a solo project but I'm open to contributions, especially:

- Additional language support (currently strongest in CJK + European)
- Alternative embedding models (faster, smaller, or better multilingual coverage)
- GPU index optimizations
- Benchmarks on different hardware

## Development

```bash
git clone https://github.com/Protonhash/sieve.git
pip install -e ".[dev]"
```

## Testing

```bash
pytest test/ -v
```

Tests use small fixture datasets (in `test/fixtures/`). No large downloads needed.

## PR guidelines

- Benchmark before/after if your change affects performance
- Document any new config options in `conf/default.yaml`
- Keep the offline-first design — no external API dependencies in core pipeline

## Architecture decisions

See the commit history and code comments for reasoning behind design choices. If you want to discuss a major change, open an issue first.
