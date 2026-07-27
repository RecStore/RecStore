# RankMixer Model

This directory hosts the RankMixer dense compute model used by the
`rs_demo` benchmark harness.

The actual model implementation (`rankmixer_model.py`, `verify_correctness.py`,
`analyze_bench.py`, benchmark notes) is **not included** in the public
repository.  These files are listed in `.gitignore` and must be obtained
from the internal team.

## Integration point

When present, the model is consumed by
`model_zoo/rs_demo/runtime/hybrid_dlrm.py` via:

```python
from model_zoo.rankmixer.rankmixer_model import build_rankmixer_arch
```

The `--model rankmixer` flag in `rs_demo` selects this architecture instead
of the default DLRM.  Without the local files the flag has no effect.

## Expected files (untracked)

| File | Description |
|------|-------------|
| `rankmixer_model.py` | RankMixer architecture (MaskBlock, TokenMixer, PFFN, PLE) |
| `verify_correctness.py` | Numerical correctness verification |
| `analyze_bench.py` | Benchmark result analysis |
| `BENCHMARK.md` | Single-node benchmark notes |
| `3NODE_BENCHMARK.md` | 3-node benchmark notes |
