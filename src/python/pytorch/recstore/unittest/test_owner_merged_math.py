"""Merged owner CUDA kernel vs CPU scalar FMA, bitwise.

Runs on a real CUDA device (SM80 A100). The oracle reproduces the kernel's
fixed order: s = (rank0 row if present else +0.0) + (rank1 row if present
else +0.0), g = s * 0.5 (the v1/v2 TW/DP world mean; halving is an exact FP
operation), updated = fmaf(-lr, g, old). Positive zero is added for absent
ranks exactly as the kernel does.
"""
import struct
import unittest

import torch

from ..ps_pipeline import owner_math

CUDA = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SUPPORTED = (CUDA.type == 'cuda'
             and torch.cuda.get_device_capability(CUDA) == (8, 0))


def cpu_f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


@unittest.skipUnless(SUPPORTED, 'requires an SM80 CUDA device')
class TestOwnerMergedMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owner_math.initialize_owner_merged_math(CUDA)

    def run_kernel(self, meta_rows, v0, v1, weights, lr):
        meta = torch.tensor(meta_rows, dtype=torch.int64, device=CUDA)
        result = owner_math._EXTENSION.owner_merged_math(
            meta, v0.to(CUDA), v1.to(CUDA), weights.to(CUDA), lr)
        torch.cuda.synchronize(CUDA)
        return result

    def oracle(self, meta_rows, v0, v1, weights, lr):
        gradient, candidate = [], []
        for row, m in enumerate(meta_rows):
            for col in range(128):
                a = cpu_f32(v0[m[4], col]) if m[5] else 0.0
                b = cpu_f32(v1[m[6], col]) if m[7] else 0.0
                # IEEE single-precision round-to-nearest addition, then the
                # exact world-2 halving (matches __fmul_rn(sum, 0.5f)).
                g = cpu_f32(cpu_f32(a + b) * 0.5)
                old = cpu_f32(weights[row, col])
                updated = cpu_f32(-cpu_f32(lr) * g + old)
                gradient.append(g)
                candidate.append(updated)
        return (torch.tensor(gradient, dtype=torch.float32).view(len(meta_rows), 128),
                torch.tensor(candidate, dtype=torch.float32).view(len(meta_rows), 128))

    def check(self, meta_rows, v0, v1, weights, lr):
        candidate, error = self.run_kernel(meta_rows, v0, v1, weights, lr)
        _, expect_c = self.oracle(meta_rows, v0, v1, weights, lr)
        self.assertEqual(error.item(), 0)
        # Compare raw FP32 bits so signed zeros cannot pass as equal.
        self.assertTrue(torch.equal(candidate.cpu().view(torch.int32), expect_c.view(torch.int32)))

    def test_presence_combinations_match_cpu_scalar_fma_bitwise(self):
        meta = [
            [10, 3, 0, 0, 0, 1, 0, 0],   # only rank 0
            [20, 0, 2, 1, 0, 0, 0, 1],   # only rank 1
            [30, 1, 1, 0, 0, 1, 0, 1],   # both ranks, offsets 0/0
            [40, 2, 1, 1, 1, 1, 0, 1],   # both ranks, nonzero offsets
            [50, 0, 5, 0, 0, 0, 2, 1],   # rank 1 only, offset 2
        ]
        v0 = torch.randn(2, 128, dtype=torch.float32, generator=torch.Generator().manual_seed(7))
        v1 = torch.randn(3, 128, dtype=torch.float32, generator=torch.Generator().manual_seed(8))
        weights = torch.randn(5, 128, dtype=torch.float32,
                              generator=torch.Generator().manual_seed(9))
        self.check(meta, v0, v1, weights, 0.01)

    def test_signed_zero_and_subnormal_inputs(self):
        rows = 3
        v0 = torch.zeros(1, 128, dtype=torch.float32)
        v0[0, 0] = -0.0
        v0[0, 1] = 1e-40          # subnormal
        v0[0, 2] = -1e-40         # negative subnormal
        v1 = torch.zeros(2, 128, dtype=torch.float32)
        v1[0, 0] = 0.0
        v1[0, 1] = 0.0
        v1[1, 2] = -0.0
        v1[1, 3] = 1e-40
        weights = torch.zeros(rows, 128, dtype=torch.float32)
        weights[1, 1] = -0.0
        meta = [
            [1, 1, 1, 0, 0, 1, 0, 1],
            [2, 0, 1, 1, 0, 0, 0, 1],
            [3, 1, 0, 0, 0, 1, 0, 0],
        ]
        self.check(meta, v0, v1, weights, 0.5)

    def test_error_code_two_on_nonfinite_arithmetic(self):
        meta = [[1, 1, 1, 0, 0, 1, 0, 1]]
        v0 = torch.full((1, 128), 3e38, dtype=torch.float32)
        v1 = torch.full((1, 128), 3e38, dtype=torch.float32)
        weights = torch.zeros(1, 128, dtype=torch.float32)
        _, error = self.run_kernel(meta, v0, v1, weights, .01)
        self.assertEqual(error.item(), 2)  # a+b overflows to inf

    def test_error_code_one_on_malformed_metadata(self):
        weights = torch.zeros(1, 128, dtype=torch.float32)
        v0, v1 = torch.zeros(1, 128, dtype=torch.float32), torch.zeros(1, 128, dtype=torch.float32)
        bad_rows = [
            [1, 1, 1, 0, 0, 2, 0, 1],    # nvec0 > 1
            [1, 1, 1, 0, 0, 1, 0, 2],    # nvec1 > 1
            [1, 0, 0, 0, 0, 0, 0, 0],    # both counts zero
            [1, 1, 1, 2, 0, 1, 0, 1],    # type out of range
            [1, 1, 1, 0, 5, 1, 0, 1],    # offset0 beyond total0
            [1, 1, 1, 0, 0, 1, 5, 1],    # offset1 beyond total1
            [1, 0, 1, 0, 0, 1, 0, 1],    # count0 zero but nvec0 one
        ]
        for row in bad_rows:
            with self.subTest(row=row):
                _, error = self.run_kernel([row], v0, v1, weights, 0.01)
                self.assertEqual(error.item(), 1)

    def test_host_checks_reject_bad_shapes_and_rates(self):
        meta = torch.zeros(1, 8, dtype=torch.int64, device=CUDA)
        v = torch.zeros(1, 128, dtype=torch.float32, device=CUDA)
        for args in ((meta.view(8), v, v[:0], v, .01),
                     (meta, torch.zeros(1, 64, device=CUDA), v[:0], v, .01),
                     (meta, v, v[:0], torch.zeros(2, 128, device=CUDA), .01),
                     (meta, v, v[:0], v, 0.0),
                     (meta, v, v[:0], v, float('nan'))):
            with self.subTest(args=args[:2]):
                with self.assertRaises(RuntimeError):
                    owner_math._EXTENSION.owner_merged_math(*args)

    def test_empty_meta_returns_empty_outputs(self):
        meta = torch.zeros(0, 8, dtype=torch.int64, device=CUDA)
        v = torch.zeros(0, 128, dtype=torch.float32, device=CUDA)
        weights = torch.zeros(0, 128, dtype=torch.float32, device=CUDA)
        candidate, error = owner_math._EXTENSION.owner_merged_math(
            meta, v, v, weights, .01)
        self.assertEqual(candidate.shape, (0, 128))
        self.assertEqual(error.item(), 0)


if __name__ == '__main__':
    unittest.main()
