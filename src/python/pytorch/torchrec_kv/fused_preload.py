"""Pre-create embedding rows in the key space the fused path actually reads.

With fusion enabled, every feature is read, written and updated through the
first (master) table, so a row with id ``id`` of feature ``i`` lives at
``(i << fusion_k) | id`` under the master table's tag. Pre-creating that row
under the feature's own table name stores it under a different tag, where the
fused path never looks: the row is allocated and written, but never found, and
the first update that touches the id pays the insert path again.

The helper stays deliberately small. It only relies on the part of the client
interface both clients share: ``register_tensor_meta`` plus ``push`` when the
client can write into the master key space.
"""

import warnings
from typing import Any, Optional

import torch


def preload_fused_rows(
    kv_client: Any,
    *,
    master_name: str,
    name: str,
    num_embeddings: int,
    embedding_dim: int,
    base_offset: int,
    values: Optional[torch.Tensor] = None,
    write_values: bool = True,
) -> None:
    """Create one feature's rows at ``base_offset`` inside the master table.

    Args:
        kv_client: client that owns the backend tables.
        master_name: table name the fused read/write path uses.
        name: the feature's own table name, registered for metadata only.
        num_embeddings: number of rows to create for this feature.
        embedding_dim: row width.
        base_offset: fused offset of this feature, ``idx << fusion_k``.
        values: optional initial rows; defaults to zeros.
        write_values: set to False to only register metadata.

    Raises:
        RuntimeError: when custom initial values are requested but the client
            cannot write into the master key space.
    """
    rows = int(num_embeddings)
    dim = int(embedding_dim)
    if rows <= 0 or dim <= 0:
        return

    kv_client.register_tensor_meta(
        name=name,
        shape=(rows, dim),
        dtype=torch.float32,
        base_offset=base_offset,
    )
    # Keep registering the feature table with the backend even though the rows
    # go into the master table. Non-initializer ranks open their PS connection
    # through this call, so skipping it here would make table setup asymmetric
    # across ranks on the RDMA backend.
    init_table = getattr(kv_client, "init_embedding_table", None)
    if callable(init_table):
        ok = init_table(name, rows, dim)
        if ok is False:
            raise RuntimeError(
                f"Failed to initialize embedding table '{name}' on backend."
            )
    if not write_values:
        return

    push = getattr(kv_client, "push", None)
    if push is None:
        if values is not None:
            raise RuntimeError(
                f"kv_client {type(kv_client).__name__} has no push(); it cannot "
                "write custom initial values into the fused key space."
            )
        # Without push the rows stay lazy: the first update creates them.
        # That keeps the old behavior instead of failing the whole job.
        warnings.warn(
            f"kv_client {type(kv_client).__name__} has no push(); fused rows "
            f"for table '{name}' are registered but not pre-created.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if values is None:
        initial = torch.zeros((rows, dim), dtype=torch.float32)
    else:
        initial = torch.as_tensor(values, dtype=torch.float32).detach()
        if initial.device.type != "cpu":
            initial = initial.to("cpu")
        if not initial.is_contiguous():
            initial = initial.contiguous()
        if tuple(initial.shape) != (rows, dim):
            raise ValueError(
                f"initial values for table '{name}' have shape "
                f"{tuple(initial.shape)}, expected {(rows, dim)}."
            )

    ids = torch.arange(rows, dtype=torch.int64) + int(base_offset)
    push(master_name, ids, initial)
