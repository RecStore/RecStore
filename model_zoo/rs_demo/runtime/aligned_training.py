from __future__ import annotations

def build_dense_stack(torch, input_dim: int):
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )


def sync_device(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare_dense_input(dense_batch, pooled_source, labels_batch, torch, device):
    sync_device(torch, device)
    pooled = pooled_source.to(device).detach().requires_grad_(True)
    dense_input = torch.cat([dense_batch.to(device), pooled], dim=1)
    labels = labels_batch.to(device).float()
    if labels.ndim == 1:
        labels = labels.view(-1, 1)
    sync_device(torch, device)
    return dense_input, pooled, labels


def run_dense_backward(loss, pooled, dense_module, torch, device):
    sync_device(torch, device)
    dense_params = [param for param in dense_module.parameters() if param.requires_grad]
    grads = torch.autograd.grad(
        loss,
        dense_params + [pooled],
        retain_graph=True,
    )
    for param, grad in zip(dense_params, grads[:-1]):
        param.grad = None if grad is None else grad.detach()
    pooled_grad = grads[-1]
    if pooled_grad is None:
        raise RuntimeError("missing pooled gradient after dense backward")
    sync_device(torch, device)
    return pooled_grad.detach()
