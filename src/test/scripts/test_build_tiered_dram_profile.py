import unittest

import numpy as np

from tools.benchmarks.build_tiered_dram_profile import build_profile


class TieredDramProfileTest(unittest.TestCase):
    def test_historical_ranking_and_ps_namespace(self):
        sparse = np.array([[0, 0], [0, 1], [1, 2], [3, 3]], dtype=np.int64)
        common = dict(cards=[4, 4], profile_end=3, dram_rows=5)
        frequency = build_profile(sparse, **common, strategy='frequency', key_encoding='contiguous')
        np.testing.assert_array_equal(frequency, [0, 1, 4, 5, 6])
        density = build_profile(sparse, **common, strategy='density', key_encoding='contiguous')
        np.testing.assert_array_equal(density, [0, 1, 4, 5, 7])
        encoded = build_profile(sparse, **common, strategy='density', key_encoding='fused', fusion_k=30, tensor_tag=3)
        tag = 3 << 56
        np.testing.assert_array_equal(encoded, [tag, tag + 1, tag + (1 << 30), tag + (1 << 30) + 1, tag + (1 << 30) + 3])
        # A later, unseen evaluation window cannot influence the selection.
        sparse[3] = [2, 0]
        np.testing.assert_array_equal(density, build_profile(sparse, **common, strategy='density', key_encoding='contiguous'))
        with self.assertRaises(ValueError):
            build_profile(sparse, **common, strategy='density', key_encoding='fused', fusion_k=1)


if __name__ == '__main__':
    unittest.main()
