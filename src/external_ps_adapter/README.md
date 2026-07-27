# External PS Adapter

This directory hosts a DRAM Index + flat ValueStore adapter designed to
integrate RecStore's lookup performance into an external recommendation
training framework.

The actual adapter implementation (`ps_store.h`, `ps_store.cc`, `row_codec.h`,
`CMakeLists.txt`) is **not included** in the public repository.  These files
are listed in `.gitignore` and must be obtained from the internal team.

## Integration model

- **LookUp hot path**: served by the adapter's flat DRAM rows + id index.
- **PushGrad / Step / Dump / Load**: reuse the external framework's existing
  Hashtable + serializer on a shadow table.
- After each trainable LookUp, embeddings are synced into the shadow Hashtable.
- After Step / Load, shadow state is rebuilt back into the adapter for the
  next lookup.

This keeps the external framework's model / RPC / checkpoint formats
unchanged while allowing lookup performance experiments.

## Build (when files are present)

```bash
USE_RECSTORE_PS=1 pip install -e .
```

## Expected files (untracked)

| File | Description |
|------|-------------|
| `ps_store.h` | Adapter class declaration + table/lookup config structs |
| `ps_store.cc` | DRAM Index + flat ValueStore implementation |
| `row_codec.h` | Row layout helpers |
| `CMakeLists.txt` | Builds a static library |
