#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

// Merged-gradient owner arithmetic. Native code validates merged records
// before upload. Metadata columns: key,c0,c1,type,offset0,nvec0,offset1,nvec1.
// Each row carries at most one merged vector per source rank (nvec in {0,1}).
// The kernel contract mirrors owner_math.cu v1: defensive device bounds, the
// same atomicOr error codes (1 malformed metadata, 2 arithmetic), optional
// gradient output, and the status[1] tensor checked by the v4 commit callback.
namespace {
__global__ void owner_merged_kernel(const int64_t* meta, const float* v0,
                                    const float* v1, const float* old_weights,
                                    int64_t total0, int64_t total1, float lr,
                                    float* gradient, float* candidate, int* error) {
  const int64_t row = blockIdx.x;
  const int col = threadIdx.x;
  const auto* m = meta + row * 8;
  // Defensive device bounds. The complete metadata contract is checked on CPU.
  if (m[1] < 0 || m[1] > 1024 || m[2] < 0 || m[2] > 1024 ||
      m[1] + m[2] == 0 || m[3] < 0 || m[3] > 1 ||
      m[4] < 0 || m[5] < 0 || m[4] > total0 || m[5] > total0-m[4] ||
      m[6] < 0 || m[7] < 0 || m[6] > total1 || m[7] > total1-m[6] ||
      (m[5] != 0 && m[5] != 1) || (m[7] != 0 && m[7] != 1) ||
      bool(m[1]) != bool(m[5]) || bool(m[2]) != bool(m[7])) {
    if (!col) atomicOr(error, 1);
    return;
  }
  // Fixed order: rank-0 contribution first, then rank-1. A missing source
  // contributes positive zero, matching the v1 type-1 absent-rank behavior.
  const float a = m[5] ? v0[int64_t(m[4]) * 128 + col] : 0.0f;
  const float b = m[7] ? v1[int64_t(m[6]) * 128 + col] : 0.0f;
  const float result = __fadd_rn(a, b);
  const int64_t at = row * 128 + col;
  const float weight = old_weights[at];
  const float updated = __fmaf_rn(-lr, result, weight);
  const bool failed = !isfinite(a) || !isfinite(b) || !isfinite(result) ||
      !isfinite(weight) || !isfinite(updated);
  if (gradient) gradient[at] = result;
  candidate[at] = updated;
  if (failed) atomicOr(error, 2);
}
}  // namespace

std::vector<torch::Tensor> owner_merged_math(torch::Tensor meta, torch::Tensor v0,
    torch::Tensor v1, torch::Tensor weights, double learning_rate, bool keep_gradient) {
  TORCH_CHECK(meta.is_cuda() && meta.scalar_type() == torch::kInt64 && meta.is_contiguous());
  TORCH_CHECK(meta.dim() == 2 && meta.size(1) == 8 && meta.size(0) <= 2*26*1024);
  for (const auto& value : {v0, v1, weights}) {
    TORCH_CHECK(value.device() == meta.device() && value.scalar_type() == torch::kFloat32);
    TORCH_CHECK(value.is_contiguous() && value.dim() == 2 && value.size(1) == 128);
  }
  TORCH_CHECK(weights.size(0) == meta.size(0) && v0.size(0)+v1.size(0) <= 2*26*1024);
  const float lr = static_cast<float>(learning_rate);
  TORCH_CHECK(std::isfinite(lr) && lr > 0, "finite positive FP32 learning rate required");
  c10::cuda::CUDAGuard guard(meta.device());
  auto gradient = torch::empty({keep_gradient ? meta.size(0) : 0, int64_t{128}}, weights.options());
  auto candidate = torch::empty_like(weights);
  auto error = torch::zeros({1}, meta.options().dtype(torch::kInt32));
  if (meta.size(0)) {
    owner_merged_kernel<<<meta.size(0),128,0,at::cuda::getCurrentCUDAStream()>>>(
        meta.data_ptr<int64_t>(), v0.data_ptr<float>(), v1.data_ptr<float>(),
        weights.data_ptr<float>(), v0.size(0), v1.size(0), lr,
        keep_gradient ? gradient.data_ptr<float>() : nullptr,
        candidate.data_ptr<float>(), error.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {gradient, candidate, error};
}
