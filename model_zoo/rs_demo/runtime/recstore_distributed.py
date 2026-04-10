from __future__ import annotations

import ctypes
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ShardServer:
    shard: int
    host: str
    port: int


def _load_runtime_config(runtime_dir: Path) -> dict:
    cfg_path = runtime_dir / "recstore_config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_client_configs(cfg: dict) -> tuple[dict, dict]:
    distributed = cfg.get("distributed_client")
    cache_ps = cfg.get("cache_ps")
    return (
        distributed if isinstance(distributed, dict) else {},
        cache_ps if isinstance(cache_ps, dict) else {},
    )


def _load_shard_servers(distributed_cfg: dict, cache_cfg: dict, cfg_path: Path) -> list[ShardServer]:
    if "servers" in distributed_cfg:
        raw_servers = distributed_cfg.get("servers") or []
    else:
        raw_servers = cache_cfg.get("servers") or []
    if not raw_servers:
        raise RuntimeError(f"missing shard servers in {cfg_path}")

    servers: list[ShardServer] = [
        ShardServer(
            shard=int(server["shard"]),
            host=str(server["host"]),
            port=int(server["port"]),
        )
        for server in raw_servers
    ]
    servers.sort(key=lambda s: s.shard)
    return servers


def _load_num_shards(distributed_cfg: dict, cache_cfg: dict, servers: list[ShardServer]) -> int:
    if "num_shards" in distributed_cfg:
        return int(distributed_cfg["num_shards"])
    if "num_shards" in cache_cfg:
        return int(cache_cfg["num_shards"])
    return len(servers)


def _load_hash_method(distributed_cfg: dict) -> str:
    return str(distributed_cfg.get("hash_method", "city_hash"))


_CITYHASH_LIB_HANDLE: ctypes.CDLL | None = None
_CITYHASH64_FUNC = None


def _get_cityhash64_func():
    global _CITYHASH_LIB_HANDLE, _CITYHASH64_FUNC
    if _CITYHASH64_FUNC is not None:
        return _CITYHASH64_FUNC
    lib_path = Path(__file__).resolve().parents[3] / "third_party/cityhash/src/.libs/libcityhash.so.0.0.0"
    try:
        _CITYHASH_LIB_HANDLE = ctypes.CDLL(str(lib_path))
    except OSError as exc:
        raise RuntimeError(f"failed to load cityhash library: {lib_path}") from exc
    try:
        func = _CITYHASH_LIB_HANDLE._Z10CityHash64PKcm
    except AttributeError as exc:
        raise RuntimeError(f"missing CityHash64 symbol in library: {lib_path}") from exc
    func.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    func.restype = ctypes.c_uint64
    _CITYHASH64_FUNC = func
    return _CITYHASH64_FUNC


def _city_hash64_of_uint64(key: int) -> int:
    raw = struct.pack("<Q", int(key) & 0xFFFFFFFFFFFFFFFF)
    city_hash64 = _get_cityhash64_func()
    return int(city_hash64(raw, len(raw)))


class ShardedRecstoreClient:
    def __init__(self, client, runtime_dir: Path) -> None:
        cfg = _load_runtime_config(runtime_dir)
        distributed_cfg, cache_cfg = _load_client_configs(cfg)
        self._client = client
        self._servers = _load_shard_servers(
            distributed_cfg, cache_cfg, runtime_dir / "recstore_config.json"
        )
        self._servers_by_shard = {server.shard: server for server in self._servers}
        self._num_shards = _load_num_shards(distributed_cfg, cache_cfg, self._servers)
        self._hash_method = _load_hash_method(distributed_cfg)
        self._active_shard: int | None = None
        self._next_prefetch_id = 1
        self._prefetch_contexts: dict[int, tuple[int, list[tuple[int, torch.Tensor, torch.Tensor]]]] = {}

    def _shard_for_key(self, key: int) -> int:
        if self._hash_method == "simple_mod":
            return int(key) % self._num_shards
        return _city_hash64_of_uint64(int(key)) % self._num_shards

    def _activate_shard(self, shard: int) -> None:
        shard = int(shard)
        if self._active_shard == shard:
            return
        server = self._servers_by_shard.get(shard)
        if server is None:
            raise RuntimeError(f"no server configured for shard {shard}")
        self._client.ops.set_ps_config(server.host, server.port)
        self._active_shard = shard

    def _group_indices(self, keys: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        shard_to_indices: dict[int, list[int]] = {}
        shard_order: list[int] = []
        for idx, key in enumerate(keys.tolist()):
            shard = self._shard_for_key(int(key))
            if shard not in shard_to_indices:
                shard_to_indices[shard] = []
                shard_order.append(shard)
            shard_to_indices[shard].append(idx)
        return [
            (shard, torch.tensor(indices, dtype=torch.long))
            for shard, indices in ((shard, shard_to_indices[shard]) for shard in shard_order)
        ]

    def init_embedding_table(self, table_name: str, num_embeddings: int, embedding_dim: int) -> bool:
        ok = True
        for server in self._servers:
            self._activate_shard(server.shard)
            ok = self._client.init_embedding_table(table_name, num_embeddings, embedding_dim) and ok
        return ok

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.numel() == 0:
            return
        for shard, index_tensor in self._group_indices(keys):
            self._activate_shard(shard)
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_vals = values.index_select(0, index_tensor).contiguous()
            self._client.emb_write(shard_keys, shard_vals)

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        if keys.numel() == 0:
            return torch.empty((0, embedding_dim), dtype=torch.float32)
        out = torch.empty((keys.shape[0], embedding_dim), dtype=torch.float32)
        for shard, index_tensor in self._group_indices(keys):
            self._activate_shard(shard)
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_values = self._client.emb_read(shard_keys, embedding_dim)
            out.index_copy_(0, index_tensor, shard_values)
        return out

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        req_id = self._next_prefetch_id
        self._next_prefetch_id += 1
        if keys.numel() == 0:
            self._prefetch_contexts[req_id] = (0, [])
            return req_id

        shard_requests: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for shard, index_tensor in self._group_indices(keys):
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_requests.append((shard, index_tensor, shard_keys))
        self._prefetch_contexts[req_id] = (int(keys.shape[0]), shard_requests)
        return req_id

    def emb_wait_result(self, prefetch_id: int, embedding_dim: int) -> torch.Tensor:
        context = self._prefetch_contexts.pop(int(prefetch_id), None)
        if context is None:
            raise RuntimeError(f"unknown prefetch_id: {prefetch_id}")
        total_rows, shard_requests = context
        if total_rows == 0:
            return torch.empty((0, embedding_dim), dtype=torch.float32)
        out = torch.empty((total_rows, embedding_dim), dtype=torch.float32)
        for shard, index_tensor, shard_keys in shard_requests:
            self._activate_shard(shard)
            shard_prefetch_id = int(self._client.emb_prefetch(shard_keys))
            shard_values = self._client.emb_wait_result(shard_prefetch_id, embedding_dim)
            out.index_copy_(0, index_tensor, shard_values)
        return out

    def emb_read_prefetch(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        prefetch_id = self.emb_prefetch(keys)
        return self.emb_wait_result(prefetch_id, embedding_dim)

    def emb_update_table(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        if keys.numel() == 0:
            return
        for shard, index_tensor in self._group_indices(keys):
            self._activate_shard(shard)
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_grads = grads.index_select(0, index_tensor).contiguous()
            self._client.emb_update_table(table_name, shard_keys, shard_grads)
