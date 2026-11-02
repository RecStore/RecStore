"""Fixed-shape FP32 dense forward/backward; PS, communication, and SGD stay outside."""
from __future__ import annotations

import time

import torch
import torch.distributed as dist


class DenseGradientBuffer:
    """Fixed gradient storage for an unwrapped module with explicit mean reduction.

    The caller broadcasts initial model state across ranks and must not also run
    DDP reductions. Graph replay, reduction, and SGD use the same current stream.
    """

    def __init__(self, module, group=None):
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            raise ValueError("Create the dense module without DDP before capture")
        self.params = [p for p in module.parameters() if p.requires_grad]
        if not self.params or any(
            p.device.type != "cuda" or p.dtype != torch.float32
            or p.device != self.params[0].device or not p.is_contiguous()
            for p in self.params
        ):
            raise ValueError("Expected nonempty contiguous FP32 CUDA parameters")
        self.group = group
        self.flat = torch.empty(
            sum(p.numel() for p in self.params),
            device=self.params[0].device, dtype=torch.float32,
        )
        self.views = [chunk.view_as(p) for p, chunk in zip(
            self.params, self.flat.split([p.numel() for p in self.params])
        )]

    def bind(self):
        for param, view in zip(self.params, self.views):
            param.grad = view

    def begin_step(self):
        self.flat.zero_()
        self.bind()

    @torch.no_grad()
    def submit_reduce(self):
        """Enqueue mean reduction; wait before SGD or reusing the flat buffer."""
        if any(p.grad is not view for p, view in zip(self.params, self.views)):
            raise RuntimeError("Dense gradient storage changed before reduction")
        world_size = dist.get_world_size(self.group) if dist.is_initialized() else 1
        if world_size > 1:
            # Match DDP's pre-division, including its FP32 overflow behavior.
            self.flat.div_(world_size)
            return dist.all_reduce(self.flat, group=self.group, async_op=True)
        return None

    @torch.no_grad()
    def reduce(self):
        work = self.submit_reduce()
        if work is not None:
            work.wait()


class CapturedDlrmDense:
    """Capture deterministic dense training for fixed full/tail batch shapes.

    Inputs are dense [B, dense_features], sparse [B, num_features, embedding_dim],
    and labels [B, 1], all FP32 on the model's CUDA device. No optimizer executes
    during construction. Modules that mutate state or consume RNG are rejected.

    Returned loss and sparse gradient storage are reused on the next replay of
    that shape. A caller must consume/copy them before replay; retaining a tensor
    reference alone does not protect an asynchronous PS reader from overwrite.
    All graphs also share dense gradient storage and must run serially.
    """

    def __init__(
        self, module, criterion, gradient_buffer, *, batch_sizes,
        num_features, embedding_dim, dense_features=13,
    ):
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            raise ValueError("Capture a module created without DDP hooks")
        params = [p for p in module.parameters() if p.requires_grad]
        if len(params) != len(gradient_buffer.params) or any(
            p is not q for p, q in zip(params, gradient_buffer.params)
        ):
            raise ValueError("Gradient buffer does not belong to the dense module")
        sizes = tuple(dict.fromkeys(batch_sizes))
        if not sizes or any(type(n) is not int or n <= 0 for n in sizes):
            raise ValueError("Capture requires positive integer batch sizes")
        if any(type(n) is not int or n <= 0 for n in (num_features, embedding_dim, dense_features)):
            raise ValueError("Dense input dimensions must be positive integers")
        self.device = params[0].device
        self.gradient_buffer = gradient_buffer
        self.graphs = {}
        self.batch_sizes = sizes
        self.static_input_bytes = 0
        self.memory_stats = {
            "allocated_before_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_before_bytes": torch.cuda.memory_reserved(self.device),
        }
        initial_state = {name: value.detach().clone() for name, value in module.state_dict().items()}
        pointers = [p.data_ptr() for p in params]
        initial_grads = [p.grad for p in params]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device)
        seen = set()
        hooks = [p.register_hook(lambda grad, i=i: seen.add(i)) for i, p in enumerate(params)]
        stream = torch.cuda.Stream(device=self.device)
        start = time.perf_counter()
        try:
            for batch_size in sizes:
                dense = torch.zeros((batch_size, dense_features), device=self.device, dtype=torch.float32)
                sparse = torch.zeros(
                    (batch_size, num_features, embedding_dim), device=self.device,
                    dtype=torch.float32, requires_grad=True,
                )
                labels = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
                sparse.grad = torch.zeros_like(sparse)
                sparse_grad = sparse.grad
                self.static_input_bytes += sum(t.numel() * t.element_size() for t in (dense, sparse, labels, sparse_grad))

                def forward_backward():
                    gradient_buffer.begin_step()
                    seen.clear()
                    sparse_grad.zero_()
                    loss = criterion(module(dense, sparse), labels)
                    loss.backward()
                    return loss

                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        forward_backward()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                # Separate private pools permit full/tail replays in any order.
                with torch.cuda.graph(graph, stream=stream):
                    loss = forward_backward()
                if len(seen) != len(params):
                    raise RuntimeError("Every dense parameter must participate in backward")
                if sparse.grad is not sparse_grad or any(
                    p.grad is not view for p, view in zip(params, gradient_buffer.views)
                ):
                    raise RuntimeError("Capture replaced fixed gradient storage")
                self.graphs[batch_size] = (graph, dense, sparse, labels, loss, sparse_grad)
            torch.cuda.synchronize(self.device)
            if pointers != [p.data_ptr() for p in params]:
                raise RuntimeError("Capture changed parameter storage")
            for name, value in module.state_dict().items():
                if not torch.equal(value, initial_state[name]):
                    raise RuntimeError(f"Capture changed model state: {name}")
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
                cuda_rng, torch.cuda.get_rng_state(self.device)
            ):
                raise RuntimeError("Dense capture requires a model that does not consume RNG")
        finally:
            for hook in hooks:
                hook.remove()
            # Warmup/capture must not change eager state, including existing grads.
            module.load_state_dict(initial_state)
            for param, grad in zip(params, initial_grads):
                param.grad = grad
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
        torch.cuda.synchronize(self.device)
        self.capture_ms = (time.perf_counter() - start) * 1000
        del initial_state
        self.memory_stats.update({
            "allocated_after_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_after_bytes": torch.cuda.memory_reserved(self.device),
            "static_input_bytes": self.static_input_bytes,
            "dense_gradient_buffer_bytes": gradient_buffer.flat.numel() * gradient_buffer.flat.element_size(),
            # Device-wide high-water marks; do not reset another caller's stats.
            "device_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "device_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        })

    def __call__(self, dense, sparse, labels):
        if dense.ndim != 2 or dense.shape[0] not in self.graphs:
            raise ValueError("Input batch size has no captured graph")
        graph, static_dense, static_sparse, static_labels, loss, sparse_grad = self.graphs[dense.shape[0]]
        inputs = ((dense, static_dense), (sparse, static_sparse), (labels, static_labels))
        for actual, expected in inputs:
            if actual.shape != expected.shape or actual.dtype != torch.float32 or actual.device != self.device:
                raise ValueError("Input shape, FP32 dtype, or CUDA device differs from capture")
        # Replay runs captured zeroing; Python gradient hooks do not run again.
        self.gradient_buffer.bind()
        with torch.no_grad():
            for actual, expected in inputs:
                expected.copy_(actual)
        graph.replay()
        return loss.detach(), sparse_grad
