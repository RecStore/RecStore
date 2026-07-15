import time
from dataclasses import dataclass

import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from .config import configure_src_paths

configure_src_paths()

from python.pytorch.recstore.KVClient import get_kv_client
from python.pytorch.recstore.optimizer import SparseSGD
from python.pytorch.torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection


LEARNING_RATE = 0.01
NUM_TEST_ROUNDS = 10
DEFAULT_FEATURE_NAME = "feature_0"


@dataclass(frozen=True)
class PrecisionRunResult:
    rounds_completed: int
    success: bool


def get_eb_configs(
    num_embeddings: int,
    embedding_dim: int,
    table_name: str = "default",
) -> list[EmbeddingBagConfig]:
    return [
        EmbeddingBagConfig(
            name=table_name,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[DEFAULT_FEATURE_NAME],
        )
    ]


def to_recstore_config_dict(
    configs: list[EmbeddingBagConfig],
) -> list[dict[str, object]]:
    return [
        {
            "name": config.name,
            "embedding_dim": config.embedding_dim,
            "num_embeddings": config.num_embeddings,
            "feature_names": config.feature_names,
        }
        for config in configs
    ]


def compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
    label: str,
    atol: float = 1e-6,
) -> bool:
    print(f"\n----- Comparing '{label}' -----")
    expected = expected.detach()
    actual = actual.detach()

    if expected.device.type != "cpu":
        expected = expected.cpu()
    if actual.device.type != "cpu":
        actual = actual.cpu()

    if expected.shape != actual.shape:
        print(f"FAILURE: {label} outputs have mismatched shapes.")
        print(f"  - Expected shape: {expected.shape}")
        print(f"  - Actual shape:   {actual.shape}")
        return False

    if torch.allclose(expected, actual, atol=atol):
        print(f"SUCCESS: {label} outputs are numerically aligned.")
        return True

    max_diff = (expected - actual).abs().max().item()
    print(f"FAILURE: {label} outputs are not aligned.")
    print(f"  - Max absolute difference: {max_diff:.8f}")
    print(f"  - Expected slice: {expected.flatten()[:8]}")
    print(f"  - Actual slice:   {actual.flatten()[:8]}")
    return False


def generate_random_batch(
    num_embeddings: int,
    batch_size: int,
    device: str,
) -> KeyedJaggedTensor:
    avg_len = max(1, (num_embeddings // batch_size) // 2)
    lengths = torch.randint(
        1,
        avg_len * 2,
        (batch_size,),
        device=device,
        dtype=torch.int32,
    )
    values = torch.randint(
        0,
        num_embeddings,
        (lengths.sum().item(),),
        device=device,
        dtype=torch.int64,
    )
    return KeyedJaggedTensor.from_lengths_sync(
        keys=[DEFAULT_FEATURE_NAME],
        values=values,
        lengths=lengths,
    )


def _connect_kv_client(max_retries: int = 5):
    last_error = None
    for attempt in range(max_retries):
        try:
            print(f"Attempting to connect to PS Server ({attempt + 1}/{max_retries})")
            return get_kv_client()
        except Exception as error:
            last_error = error
            if attempt < max_retries - 1:
                print(f"Connection attempt {attempt + 1} failed: {error}")
                time.sleep(1)

    raise RuntimeError(
        f"Failed to connect to PS Server after {max_retries} attempts: {last_error}"
    )


def _register_table_metadata(kv_client, table_name: str, shape: tuple[int, int]) -> None:
    kv_client._tensor_meta[table_name] = {"shape": shape, "dtype": torch.float32}
    kv_client._full_data_shape[table_name] = shape
    kv_client._data_name_list.add(table_name)
    kv_client._gdata_name_list.add(table_name)


def initialize_backend_from_standard_ebc(
    kv_client,
    standard_ebc: EmbeddingBagCollection,
    config: EmbeddingBagConfig,
) -> torch.Tensor:
    with torch.no_grad():
        initial_weights = standard_ebc.state_dict()[
            f"embedding_bags.{config.name}.weight"
        ]
        if initial_weights.device.type != "cpu":
            initial_weights = initial_weights.cpu()
        initial_weights = initial_weights.contiguous().clone()

    success = kv_client.ops.init_embedding_table(
        config.name,
        int(config.num_embeddings),
        int(config.embedding_dim),
    )
    if not success:
        print(f"Warning: init_embedding_table returned False for '{config.name}'")

    all_keys = torch.arange(config.num_embeddings, dtype=torch.int64)
    kv_client.ops.emb_write(all_keys, initial_weights)
    _register_table_metadata(
        kv_client,
        config.name,
        (config.num_embeddings, config.embedding_dim),
    )
    return initial_weights


def run_precision(args) -> PrecisionRunResult:
    if getattr(args, "ps_host", None) and getattr(args, "ps_port", None):
        get_kv_client().set_ps_config(args.ps_host, args.ps_port)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Running EBC precision baseline on {device}")

    eb_configs = get_eb_configs(args.num_embeddings, args.embedding_dim)
    config = eb_configs[0]
    standard_ebc = EmbeddingBagCollection(tables=eb_configs, device=device)

    kv_client = _connect_kv_client()
    initial_weights = initialize_backend_from_standard_ebc(
        kv_client,
        standard_ebc,
        config,
    )

    recstore_ebc = RecStoreEmbeddingBagCollection(
        embedding_bag_configs=to_recstore_config_dict(eb_configs),
        lr=LEARNING_RATE,
        enable_fusion=False,
    ).to(device)

    all_keys = torch.arange(config.num_embeddings, dtype=torch.int64)
    with torch.no_grad():
        pulled_weights = kv_client.pull(name=config.name, ids=all_keys)
        if not compare_tensors(
            initial_weights,
            pulled_weights,
            "Initial Weight Synchronization",
        ):
            raise AssertionError("Initial weight synchronization failed")

    standard_optimizer = torch.optim.SGD(standard_ebc.parameters(), lr=LEARNING_RATE)
    sparse_optimizer = SparseSGD([recstore_ebc], lr=LEARNING_RATE)

    for round_index in range(NUM_TEST_ROUNDS):
        print(f"\n### Starting Test Round {round_index + 1} of {NUM_TEST_ROUNDS} ###")
        batch = generate_random_batch(args.num_embeddings, args.batch_size, device)

        standard_output = standard_ebc(batch)
        recstore_output = recstore_ebc(batch)
        if not compare_tensors(
            standard_output.values(),
            recstore_output.values(),
            f"Round {round_index + 1} Forward Pass",
        ):
            raise AssertionError(f"Forward pass failed in round {round_index + 1}")

        standard_optimizer.zero_grad()
        sparse_optimizer.zero_grad()

        standard_output.values().sum().backward()
        recstore_output.values().sum().backward()

        if len(recstore_ebc._trace) == 0:
            raise AssertionError("RecStore EBC trace is empty after backward pass")

        standard_optimizer.step()
        sparse_optimizer.step()
        sparse_optimizer.flush()

        with torch.no_grad():
            updated_standard_weights = standard_ebc.state_dict()[
                f"embedding_bags.{config.name}.weight"
            ]
            updated_recstore_weights = kv_client.pull(name=config.name, ids=all_keys)
            if not compare_tensors(
                updated_standard_weights,
                updated_recstore_weights,
                f"Round {round_index + 1} Updated Weights",
            ):
                raise AssertionError(
                    f"Weight update check failed in round {round_index + 1}"
                )

    print(f"All {NUM_TEST_ROUNDS} precision test rounds passed.")
    return PrecisionRunResult(rounds_completed=NUM_TEST_ROUNDS, success=True)
