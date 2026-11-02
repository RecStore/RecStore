"""Merged-gradient world2 owner arithmetic; one batch of GPU staging.

The exchange, validation, staging and cleanup contracts are inherited from
GPUOwnerExchange/ForwardGPUOwnerExchange unchanged. Only the arithmetic hooks
are replaced: per row g = (rank0 row?) + (rank1 row?) in fixed order, then
updated = fmaf(-lr, g, old) on the v2 kernel (recstore_owner_math_v2, its own
JIT cache name, same strict FP flags as v1).
"""
from pathlib import Path

import torch

from .owner_gpu import GPUOwnerExchange
from .owner_forward_gpu import ForwardGPUOwnerExchange

_EXTENSION = None
_WARMED_DEVICES = set()


def _cuda_device(device):
    if isinstance(device, bool):
        raise ValueError('Merged owner arithmetic requires a CUDA device')
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('Merged owner arithmetic requires a CUDA device')
    return torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)


def initialize_owner_merged_math(device):
    """JIT and warm up before training/timing; never called implicitly per step."""
    from torch.utils.cpp_extension import load_inline
    global _EXTENSION
    device = _cuda_device(device)
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError('Merged owner arithmetic is validated only on SM80')
    if _EXTENSION is None:
        _EXTENSION = load_inline(
            name='recstore_owner_math_v2',
            cpp_sources='std::vector<torch::Tensor> owner_merged_math(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, bool);',
            cuda_sources=Path(__file__).with_name('owner_merged_math.cu').read_text(),
            functions=['owner_merged_math'], extra_cflags=['-O3', '-fno-fast-math', '-ffp-contract=off'],
            extra_cuda_cflags=['-O3', '--ftz=false', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'],
            verbose=False)
    if device.index not in _WARMED_DEVICES:
        with torch.cuda.device(device):
            meta = torch.tensor([[0, 1, 0, 0, 0, 1, 0, 0]], dtype=torch.int64, device=device)
            values = torch.zeros((1, 128), dtype=torch.float32, device=device)
            _EXTENSION.owner_merged_math(meta, values, values[:0], values, .01, False)
            torch.cuda.synchronize(device)
        _WARMED_DEVICES.add(device.index)


class MergedGPUOwnerExchange(GPUOwnerExchange):
    """Same plan/exchange contract; merged-row kernel arithmetic."""

    def _check_math_ready(self):
        if _EXTENSION is None or self.device.index not in _WARMED_DEVICES:
            raise RuntimeError('Call initialize_owner_merged_math before training')

    def _owner_math(self, meta, v0, v1, weights, learning_rate):
        return _EXTENSION.owner_merged_math(meta, v0, v1, weights, float(learning_rate), False)


class ForwardMergedGPUOwnerExchange(ForwardGPUOwnerExchange, MergedGPUOwnerExchange):
    """Unified forward staging with merged-row backward arithmetic."""
