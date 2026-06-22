# RecStore Quantarec PS Adapter

Self-contained DRAM Index + flat ValueStore adapter for Quantarec Training PS.
Designed to compile into Quantarec without pulling the full RecStore third_party tree.

## Layout

- `quantarec_ps_store.h/.cc` — table reset, lookup, insert, stats
- `quantarec_row_codec.h` — row layout helpers

## Build inside Quantarec

```bash
USE_RECSTORE_PS=1 pip install -e .
```

## Runtime

```bash
export QUANTAREC_PS_STORAGE_BACKEND=recstore
python3 test/start_training_ps.py --port 18010
```

Default backend remains `hashtable`.
