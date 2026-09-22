# BagPipe GPU cache ownership

BagPipe mode treats the C++ GPU cache as owned storage. The Python controller
may plan prefetches, but it never assumes that a replace-style insert succeeded
or that an invalidation removed a row.

## State ownership

The C++ hash table is the residency authority. The Python flat tensors are a
mirror that is updated only from authoritative C++ results:

- `prefill_gpu_cache_no_evict` returns a per-key insert mask. Only true rows
  are marked cached.
- `gpu_cache_lookup_flat_no_evict` returns values and a post-fill residency
  mask. Failed inserts remain uncached and are retried later.
- `invalidate_gpu_cache_with_mask` returns a per-key removal mask. Only rows
  reported as removed are cleared from the Python mirror.
- `contains_gpu_cache` validates a policy all-hit decision immediately before
  the lock-free `gpu_cache_lookup_flat_assuming_hits` path.
- `get_gpu_cache_generation` changes whenever the C++ cache is reset. BagPipe
  fails loudly if another component resets its owned cache.

The mirror therefore has one writer protocol:

```text
C++ no-evict insert success -> cached=true, TTL refresh
C++ lookup fill success    -> cached=true, TTL refresh
C++ removal success        -> cached=false, dirty=false, TTL=0
C++ insert/removal failure -> previous state is retained
```

## Eviction

Only BagPipe evicts rows. C++ no-evict fills never replace a resident key.
Python chooses expired or low-TTL candidates, reads dirty values for PS
writeback, and invalidates them explicitly. Bookkeeping changes only for keys
that the C++ removal mask confirms. If a no-evict fill fails because hash sets
are saturated, the next cleanup retires a broader low-TTL slice and retries.

This mirrors upstream BagPipe's `BagCache` invariant: insertion fills known
empty storage, eviction is explicit, and the mapping and occupancy state change
together. RecStore keeps the physical slots inside the C++ set-associative hash
table, so the Python mirror is keyed by embedding ID rather than by local slot.

## Failure policy

BagPipe requires the authoritative APIs above. Missing APIs are configuration
errors, not reasons to fall back to replace-style fills: a fallback could evict
a key without updating the mirror. An external cache reset is also a hard error
because dirty local updates cannot be reconstructed safely after the reset.

