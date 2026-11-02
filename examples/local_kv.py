#!/usr/bin/env python3
"""Minimal self-check for the RecStore local KV Python binding."""

from tempfile import TemporaryDirectory

import numpy as np

from recstore import _recstore_kv


def main() -> None:
    ids = np.array([7, 3], dtype=np.int64)
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    with TemporaryDirectory(prefix="recstore-local-kv-") as value_path:
        store = _recstore_kv.LocalKV(
            capacity=32,
            embedding_dim=3,
            index_type="DRAM_PET_HASH",
            value_path=value_path,
        )

        np.testing.assert_array_equal(store.lookup(ids), np.zeros_like(values))

        store.put(ids, values)
        np.testing.assert_array_equal(store.lookup(ids), values)

        gradients = np.ones_like(values)
        store.sgd_update(ids, gradients, lr=0.1)
        np.testing.assert_allclose(store.lookup(ids), values - 0.1 * gradients)

    print(
        f"RecStore {_recstore_kv.RECSTORE_VERSION}: "
        "missing lookup, put/lookup, and SGD update passed"
    )


if __name__ == "__main__":
    main()
