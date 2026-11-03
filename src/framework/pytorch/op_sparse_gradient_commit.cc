#include "framework/pytorch/op_sparse_gradient_commit.h"
#include "framework/pytorch/op_internal.h"
#include "framework/common/canonical_sparse_ticket.h"
#include "framework/common/owner_merged_plan.h"

namespace recstore::framework {
namespace {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor> canonical_sparse_ticket(
    const torch::Tensor& ids, const torch::Tensor& offsets,
    const torch::Tensor& types) {
  for (const auto* tensor : {&ids, &offsets, &types}) {
    TORCH_CHECK(tensor->device().is_cpu() && tensor->scalar_type() == torch::kInt64 &&
                    tensor->is_contiguous(),
                "Canonical ticket inputs require contiguous CPU int64 tensors");
  }
  TORCH_CHECK(ids.dim() == 2 && offsets.dim() == 1 && types.dim() == 1,
              "Canonical ticket expects IDs[F,B], offsets[F], types[F]");
  const int64_t features = ids.size(0), batch = ids.size(1);
  TORCH_CHECK(features >= 1 && features <= 26 && batch >= 0 && batch <= 1024 &&
                  offsets.numel() == features && types.numel() == features,
              "Invalid canonical ticket fixed-workload dimensions");
  const auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);
  const int64_t size = features * batch;
  auto keys = torch::empty({size}, options), counts = torch::empty({size}, options);
  auto positions = torch::empty({size}, options), starts = torch::empty({size}, options);
  auto key_types = torch::empty({size}, options), inverse = torch::empty({size}, options);
  const auto unique = BuildCanonicalSparseTicket(
      ids.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), types.data_ptr<int64_t>(),
      features, batch, keys.data_ptr<int64_t>(), counts.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(), starts.data_ptr<int64_t>(),
      key_types.data_ptr<int64_t>(), inverse.data_ptr<int64_t>());
  return {keys.narrow(0, 0, unique).clone(), counts.narrow(0, 0, unique).clone(),
          positions, starts.narrow(0, 0, unique).clone(),
          key_types.narrow(0, 0, unique).clone(), inverse};
}

std::vector<torch::Tensor> owner_merged_plan_tensors(const torch::Tensor& keys0,
                     const torch::Tensor& counts0,
                     const torch::Tensor& keys1, const torch::Tensor& counts1,
                     const torch::Tensor& boundaries, const torch::Tensor& types,
                     int64_t tag, int64_t rank, bool forward) {
  for (const auto* tensor : {&keys0, &counts0, &keys1, &counts1, &boundaries, &types}) {
    TORCH_CHECK(tensor->device().is_cpu() && tensor->dim() == 1 &&
                    tensor->scalar_type() == torch::kInt64 && tensor->is_contiguous(),
                "Merged owner plan inputs require contiguous CPU int64 vectors");
  }
  TORCH_CHECK(keys0.numel() == counts0.numel() && keys1.numel() == counts1.numel() &&
                  boundaries.numel() == types.numel() + 1,
              "Merged owner plan input lengths disagree");
  auto plan = BuildOwnerMergedPlan(
      keys0.numel(), keys0.data_ptr<int64_t>(), counts0.data_ptr<int64_t>(),
      keys1.numel(), keys1.data_ptr<int64_t>(), counts1.data_ptr<int64_t>(),
      types.numel(), boundaries.data_ptr<int64_t>(), types.data_ptr<int64_t>(), tag, rank, forward);
  const auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);
  auto copy = [&](const int64_t* data, size_t n) {
    auto result = torch::empty({static_cast<int64_t>(n)}, options);
    if (n) std::memcpy(result.data_ptr<int64_t>(), data, n * sizeof(int64_t));
    return result;
  };
  std::vector<torch::Tensor> outputs{copy(plan.permutation.data(), plan.permutation.size()),
          copy(plan.keys.data(), plan.keys.size()),
          copy(plan.meta.data(), plan.meta.size()).reshape({-1, 8}),
          copy(plan.send.data(), 2), copy(plan.recv.data(), 2)};
  if (forward) {
    outputs.push_back(copy(plan.forward_send_index.data(), plan.forward_send_index.size()));
    outputs.push_back(copy(plan.forward_send.data(), 2));
    outputs.push_back(copy(plan.forward_recv.data(), 2));
    outputs.push_back(copy(plan.forward_reorder.data(), plan.forward_reorder.size()));
  }
  return outputs;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
owner_merged_plan(const torch::Tensor& keys0, const torch::Tensor& counts0,
                  const torch::Tensor& keys1, const torch::Tensor& counts1,
                  const torch::Tensor& boundaries, const torch::Tensor& types,
                  int64_t tag, int64_t rank) {
  auto outputs = owner_merged_plan_tensors(keys0, counts0, keys1, counts1,
                                           boundaries, types, tag, rank, false);
  return {outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]};
}

std::vector<torch::Tensor> owner_merged_forward_plan(
    const torch::Tensor& keys0, const torch::Tensor& counts0,
    const torch::Tensor& keys1, const torch::Tensor& counts1,
    const torch::Tensor& boundaries, const torch::Tensor& types,
    int64_t tag, int64_t rank) {
  return owner_merged_plan_tensors(keys0, counts0, keys1, counts1,
                                   boundaries, types, tag, rank, true);
}

int64_t sparse_gradient_commit_async(
    const std::string& table,
    const torch::Tensor& ids,
    const torch::Tensor& grads,
    int64_t session,
    int64_t step,
    int64_t rank,
    int64_t world,
    double lr,
    int64_t divisor) {
  TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == torch::kInt64 &&
                  grads.dim() == 2 && grads.scalar_type() == torch::kFloat32 &&
                  ids.numel() == grads.size(0),
              "Invalid sparse gradients");
  auto h = MakeSparseGradientCommitHeader(
      session, step, rank, world, grads.size(1), lr, divisor, 0);
  auto keys       = ids.is_cuda() ? StageCudaTensorToPinnedCpuAsyncNoCast(ids)
                                  : ids.contiguous().clone();
  auto keys_ready = RecordCudaCompletionForTensor(ids);
  auto values = grads.is_cuda() ? StageCudaTensorToPinnedCpuAsyncNoCast(grads)
                                : grads.contiguous().clone();
  auto grads_ready = RecordCudaCompletionForTensor(grads);
  auto client      = GetConcreteKVClientOp();
  return client->SparseGradientCommitAsync(
      table,
      h,
      ToRecTensor(keys, base::DataType::UINT64),
      ToRecTensor(values, base::DataType::FLOAT32),
      [keys, values, ids, grads, keys_ready, grads_ready] {
        keys_ready();
        grads_ready();
      });
}

int64_t sparse_owner_replace_async(
    const std::string& table, const torch::Tensor& ids,
    const torch::Tensor& candidates, const torch::Tensor& status,
    int64_t session, int64_t step, int64_t rank, int64_t world, double lr) {
  TORCH_CHECK(ids.device().is_cpu() && ids.dim() == 1 &&
                  ids.scalar_type() == torch::kInt64 &&
                  candidates.dim() == 2 && candidates.size(1) == 128 &&
                  candidates.scalar_type() == torch::kFloat32 &&
                  candidates.size(0) == ids.numel() &&
                  status.device() == candidates.device() &&
                  status.scalar_type() == torch::kInt32 &&
                  status.dim() == 1 && status.numel() == 1,
              "Owner replacement requires CPU IDs[N], FP32[N,128], and int32 status[1]");
  auto h = MakeOwnerExactReplaceHeader(session, step, rank, world, 128, lr, 0);
  auto keys = ids.contiguous().clone();
  auto values = candidates.is_cuda() ? StageCudaTensorToPinnedCpuAsyncNoCast(candidates)
                                     : candidates.contiguous().clone();
  auto checked = status.is_cuda() ? StageCudaTensorToPinnedCpuAsyncNoCast(status)
                                  : status.contiguous().clone();
  auto ready = RecordCudaCompletionForTensor(candidates);
  return GetConcreteKVClientOp()->SparseGradientCommitAsync(
      table, h, ToRecTensor(keys, base::DataType::UINT64),
      ToRecTensor(values, base::DataType::FLOAT32),
      [keys, values, checked, ids, candidates, status, ready] {
        // The event covers both D2H copies; no candidate is posted on error.
        ready();
        TORCH_CHECK(checked.data_ptr<int32_t>()[0] == 0,
                    "GPU owner rejected nonfinite or malformed candidate rows");
      });
}

void sparse_owner_replace_close(
    const std::string& table, int64_t session, int64_t step, int64_t rank,
    int64_t world, double lr, bool abort) {
  auto h = MakeOwnerExactReplaceHeader(session, step, rank, world, 128, lr, abort ? 3 : 2);
  GetConcreteKVClientOp()->CloseSparseGradientCommit(table, h);
}

void sparse_gradient_commit_wait(int64_t handle) {
  GetConcreteKVClientOp()->WaitSparseGradientCommit(handle);
}

void sparse_gradient_commit_close(
    const std::string& table,
    int64_t session,
    int64_t step,
    int64_t rank,
    int64_t world,
    int64_t dim,
    double lr,
    int64_t divisor,
    bool abort) {
  auto h = MakeSparseGradientCommitHeader(
      session, step, rank, world, dim, lr, divisor, abort ? 3 : 2);
  GetConcreteKVClientOp()->CloseSparseGradientCommit(table, h);
}
} // namespace
void RegisterSparseGradientCommitOps(torch::Library& m) {
  m.def("canonical_sparse_ticket", canonical_sparse_ticket);
  m.def("owner_merged_plan", owner_merged_plan);
  m.def("owner_merged_forward_plan", owner_merged_forward_plan);
  m.def("sparse_gradient_commit_async", sparse_gradient_commit_async);
  m.def("sparse_owner_replace_async", sparse_owner_replace_async);
  m.def("sparse_owner_replace_close", sparse_owner_replace_close);
  m.def("sparse_gradient_commit_wait", sparse_gradient_commit_wait);
  m.def("sparse_gradient_commit_close", sparse_gradient_commit_close);
}
} // namespace recstore::framework
