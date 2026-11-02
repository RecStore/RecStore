#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// One block per CPU-planned task; each lane owns one embedding coordinate.
// This proof kernel prioritizes explicit FP32 order over production throughput.
__global__ void reduce_kernel(const float* grad, const int64_t* positions,
                              const int64_t* tasks, float* output) {
  const auto* task = tasks + blockIdx.x * 6;
  const int mode = task[0], count = task[2], d = threadIdx.x;
  const int64_t start = task[1], dst = task[3];
  const bool before = task[4], after = task[5];
  if (mode == 2) {
    for (int j = 0; j < count; ++j) {
      float value = grad[positions[start+j]*128+d];
      if (before) value = __fmul_rn(value, 0.5f);
      output[(dst+j)*128+d] = value;
    }
    return;
  }
  float result = 0.0f;
  if (mode == 0) {
    for (int j = 0; j < count; ++j) {
      float value = grad[positions[start+j]*128+d];
      if (before) value = __fmul_rn(value, 0.5f);
      result = __fadd_rn(result, value);
    }
  } else {
    float leaves[32];
    const int width = (count+31)/32;
    for (int leaf = 0; leaf < 32; ++leaf) {
      float value = 0.0f;
      for (int j = leaf*width; j < min((leaf+1)*width,count); ++j) {
        float occurrence = grad[positions[start+j]*128+d];
        if (before) occurrence = __fmul_rn(occurrence, 0.5f);
        value = __fadd_rn(value, occurrence);
      }
      leaves[leaf] = value;
    }
    if (count > width)
      for (int stride = 16; stride; stride /= 2)
        for (int j = 0; j < stride; ++j)
          leaves[j] = __fadd_rn(leaves[j], leaves[j+stride]);
    result = leaves[0];
  }
  if (after) result = __fmul_rn(result, 0.5f);
  output[dst*128+d] = result;
}

torch::Tensor canonical_reduce(torch::Tensor grad, torch::Tensor positions,
                               torch::Tensor tasks, int64_t vectors) {
  TORCH_CHECK(grad.is_cuda() && grad.scalar_type()==torch::kFloat32 && grad.is_contiguous());
  TORCH_CHECK(grad.dim()==3 && grad.size(2)==128 && grad.size(0)<=1024);
  TORCH_CHECK(positions.device()==grad.device() && tasks.device()==grad.device());
  TORCH_CHECK(positions.scalar_type()==torch::kInt64 && tasks.scalar_type()==torch::kInt64);
  TORCH_CHECK(positions.is_contiguous() && tasks.is_contiguous());
  TORCH_CHECK(positions.dim()==1 && tasks.dim()==2 && tasks.size(1)==6 && vectors>=0);
  c10::cuda::CUDAGuard guard(grad.device());
  auto output = torch::empty({vectors, int64_t{128}}, grad.options());
  if (tasks.size(0)) {
    reduce_kernel<<<tasks.size(0),128,0,at::cuda::getCurrentCUDAStream()>>>(
        grad.data_ptr<float>(), positions.data_ptr<int64_t>(), tasks.data_ptr<int64_t>(),
        output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
