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
python3 test/start_training_ps.py --port 18010 --ps-storage-backend recstore
```

Default backend remains `hashtable`.

## Quantarec integration model

- **LookUp hot path**: served by `QuantarecPsStore` (flat DRAM rows + id index).
- **PushGrad / Step / Dump / Load**: reuse existing `Hashtable` + serializer on a shadow table.
- After each trainable `LookUp`, embeddings are synced into the shadow `Hashtable`.
- After `Step` / `Load`, shadow state is rebuilt back into `QuantarecPsStore` for the next lookup.

This keeps RankMixer / RPC / checkpoint formats unchanged while allowing lookup perf experiments.
