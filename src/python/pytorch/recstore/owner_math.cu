#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

// Experimental owner arithmetic. Native code validates merged records before upload.
// Metadata columns: key,c0,c1,type,offset0,nvec0,offset1,nvec1.
// Input vectors already contain the protocol's TW/DP division by two.
namespace {
struct Column {
  const float* data[2];
  int used[2], count[2], column;
  bool failed;
  __device__ Column(const float* a, const float* b, int n0, int n1, int col)
      : data{a,b}, used{0,0}, count{n0,n1}, column(col), failed(false) {}

  __device__ float take(int rank) {
    if (used[rank] >= count[rank]) { failed = true; return 0.0f; }
    float value = data[rank][int64_t(used[rank]++) * 128 + column];
    if (!isfinite(value)) failed = true;
    return value;
  }
  __device__ float add(float a, float b) {
    float value = __fadd_rn(a, b);
    if (!isfinite(value)) failed = true;
    return value;
  }
  __device__ float flush(float value) {
    unsigned bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) < 0x00800000u) bits &= 0x80000000u;
    return __uint_as_float(bits);
  }
  __device__ float atomic_style_add(float a, float b) {
    return flush(add(flush(a), flush(b)));
  }
  __device__ float leaf(int lo, int hi, int c0) {
    if (lo >= hi) return 0.0f;
    if (hi <= c0) return take(0);
    if (lo >= c0) return take(1);
    float result = take(0);
    for (int at = c0; at < hi; ++at) result = add(result, take(1));
    return result;
  }
};

__global__ void owner_kernel(const int64_t* meta, const float* v0,
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
      bool(m[1]) != bool(m[5]) || bool(m[2]) != bool(m[7])) {
    if (!col) atomicOr(error, 1);
    return;
  }
  const int c0 = m[1], n = m[1] + m[2];
  Column state(m[5] ? v0 + m[4]*128 : nullptr,
               m[7] ? v1 + m[6]*128 : nullptr, m[5], m[7], col);
  float result = 0.0f;
  if (m[3] == 1) {
    const float a = m[1] ? state.take(0) : 0.0f;
    const float b = m[2] ? state.take(1) : 0.0f;
    result = state.add(a, b);  // Preserve absent-rank positive zero.
  } else if (n < 32) {
    result = state.leaf(0, n, c0);
  } else {
    float roots[2] = {0.0f, 0.0f};
    int nr = 0;
    for (int base = 0; base < n; base += 1024) {
      const int end = min(base + 1024, n);
      float root;
      if (end <= c0) root = state.take(0);
      else if (base >= c0) root = state.take(1);
      else {
        const int width = (end-base+31)/32;
        float leaves[32];
        for (int j = 0; j < 32; ++j)
          leaves[j] = state.leaf(base+j*width, min(base+(j+1)*width, end), c0);
        for (int stride = 16; stride; stride /= 2)
          for (int j = 0; j < stride; ++j)
            leaves[j] = state.add(leaves[j], leaves[j+stride]);
        root = leaves[0];
      }
      roots[nr++] = root;
    }
    if (nr == 1) result = roots[0];
    else for (int r = 0; r < nr; ++r) result = state.atomic_style_add(result, roots[r]);
  }
  if (state.used[0] != state.count[0] || state.used[1] != state.count[1])
    state.failed = true;
  const int64_t at = row*128+col;
  const float weight = old_weights[at];
  const float updated = __fmaf_rn(-lr, result, weight);
  if (!isfinite(weight) || !isfinite(result) || !isfinite(updated)) state.failed = true;
  if (gradient) gradient[at] = result;
  candidate[at] = updated;
  if (state.failed) atomicOr(error, 2);
}
}  // namespace

std::vector<torch::Tensor> owner_math(torch::Tensor meta, torch::Tensor v0,
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
    owner_kernel<<<meta.size(0),128,0,at::cuda::getCurrentCUDAStream()>>>(
        meta.data_ptr<int64_t>(), v0.data_ptr<float>(), v1.data_ptr<float>(),
        weights.data_ptr<float>(), v0.size(0), v1.size(0), lr,
        keep_gradient ? gradient.data_ptr<float>() : nullptr,
        candidate.data_ptr<float>(), error.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {gradient, candidate, error};
}
