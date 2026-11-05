"""Optional CPU snapshots for strict learned-weight comparisons outside timing."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist


def _cpu_copy(tensor):
    return tensor.detach().to(device="cpu").clone()


def export_validation_state(path, embedding_module, dense_module, planned_controller=None):
    """Save plain CPU tensors, preserving the offsets of TorchRec local shards.

    This is a diagnostic snapshot, not a resumable training checkpoint. RecStore
    snapshots read committed authoritative values from the PS.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(dense_module, torch.nn.parallel.DistributedDataParallel):
        dense_module = dense_module.module
    state = {
        "version": 1,
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "dense": {name: _cpu_copy(value) for name, value in dense_module.state_dict().items()},
    }
    if planned_controller is not None:
        planned_controller._wait_pending()
        if callable(getattr(planned_controller, "validation_ids", None)):
            ids = planned_controller.validation_ids()
        else:
            plans = planned_controller._plans
            ids = (torch.unique(torch.cat([p.active_ids for p in plans])) if plans
                   else planned_controller._metadata.updated_ids().sort().values)
        values = torch.cat([
            planned_controller.kv_client.pull(planned_controller.master_table_name, chunk).cpu()
            for chunk in ids.split(4096)
        ]) if ids.numel() else torch.empty((0, planned_controller.embedding_dim))
        state.update(
            embedding_kind="planned",
            global_ids=ids,
            values=values,
            tables=[{
                "name": config.name,
                "offset": index << embedding_module._fusion_k,
                "shape": [config.num_embeddings, config.embedding_dim],
            } for index, config in enumerate(embedding_module._embedding_bag_configs)],
        )
    else:
        if getattr(embedding_module, "is_recstore_sparse_module", False):
            raise ValueError("RecStore validation snapshots require its read controller")
        tables = {}
        for name, value in embedding_module.state_dict().items():
            if "embedding_bags." not in name or not name.endswith(".weight"):
                raise ValueError(f"Unsupported embedding state entry: {name}")
            name = name.split("embedding_bags.", 1)[1].removesuffix(".weight")
            if name in tables:
                raise ValueError(f"Duplicate embedding table: {name}")
            if callable(getattr(value, "local_shards", None)):
                shards = [{"offsets": list(shard.metadata.shard_offsets),
                           "tensor": _cpu_copy(shard.tensor)} for shard in value.local_shards()]
            elif isinstance(value, torch.Tensor) and type(value).__name__ != "DTensor":
                shards = [{"offsets": [0, 0], "tensor": _cpu_copy(value)}]
            else:
                raise ValueError(f"Unsupported embedding state type: {type(value).__name__}")
            shape = list(value.size())
            if len(shape) != 2 or any(s["tensor"].ndim != 2 for s in shards):
                raise ValueError(f"Expected a matrix for embedding table {name}")
            tables[name] = {"shape": shape, "shards": shards}
        if not tables:
            raise ValueError("No TorchRec embedding tables found for validation")
        state.update(embedding_kind="torchrec_local_shards", tables=tables)
    temporary = path.with_name(path.name + ".partial")
    torch.save(state, temporary)
    temporary.replace(path)
    return path
