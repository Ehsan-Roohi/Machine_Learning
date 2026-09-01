import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Stage-4 tests require the Unity PyTorch environment") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(name: str):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


s3 = load("stage3_spatial")
s4 = load("stage4_pof_audit")


def mock_data(n=8):
    return s3.SpatialData(
        patches=np.zeros((n, 6, 3, 4), np.float32),
        context=np.zeros((n, 2), np.float32), targets=np.zeros((n, 3), np.float32),
        weights=np.ones(n, np.float32),
        case_id=np.asarray(["a"] * (n // 2) + ["b"] * (n - n // 2)),
        surface_i=np.arange(n), s01=np.linspace(0, 1, n),
        Ma=np.ones(n), Kn=np.ones(n), geom=np.asarray(["ISO"] * n),
        tangent=np.zeros((n, 2)), normal=np.zeros((n, 2)),
        t_grid=np.arange(4), n_grid=np.arange(3),
        target_scale_exponents=np.asarray((2, 3, 2)),
    )


class Stage4PoFAuditTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.patches = rng.normal(size=(8, 6, 3, 4)).astype(np.float32)
        self.raw = self.patches.copy()
        self.raw[:, 5] = 1.0

    def test_cell_permutation_preserves_physical_marginals_and_auxiliary_channels(self):
        transformed = s4.cell_permutation_variant(self.patches)
        self.assertTrue(np.array_equal(transformed[:, 4:], self.patches[:, 4:]))
        for sample in range(len(self.patches)):
            for channel in range(4):
                self.assertTrue(np.array_equal(
                    np.sort(transformed[sample, channel].ravel()),
                    np.sort(self.patches[sample, channel].ravel()),
                ))
        self.assertFalse(np.array_equal(transformed[:, :4], self.patches[:, :4]))

    def test_patch_mean_destroys_structure_but_keeps_auxiliary_channels(self):
        transformed = s4.valid_weighted_patch_mean(self.raw, self.patches)
        self.assertTrue(np.array_equal(transformed[:, 4:], self.patches[:, 4:]))
        for channel in range(4):
            spread = transformed[:, channel].std(axis=(1, 2))
            self.assertTrue(np.allclose(spread, 0.0, atol=1e-7))

    def test_surface_permutation_preserves_each_case_multiset(self):
        data = mock_data()
        transformed = s4.surface_permutation_variant(data, self.patches)
        for case_id in ("a", "b"):
            location = np.flatnonzero(data.case_id == case_id)
            original_rows = sorted(map(bytes, self.patches[location].copy().reshape(len(location), -1)))
            changed_rows = sorted(map(bytes, transformed[location].copy().reshape(len(location), -1)))
            self.assertEqual(original_rows, changed_rows)
        self.assertFalse(np.array_equal(transformed, self.patches))

    def test_case_pool_removes_surface_dependence_only_from_physical_channels(self):
        data = mock_data()
        transformed = s4.case_pool_variant(data, self.patches)
        for case_id in ("a", "b"):
            location = np.flatnonzero(data.case_id == case_id)
            for index in location[1:]:
                self.assertTrue(np.allclose(transformed[index, :4], transformed[location[0], :4]))
        self.assertTrue(np.array_equal(transformed[:, 4:], self.patches[:, 4:]))

    def test_channel_ablation_changes_exactly_one_channel(self):
        transformed = s4.channel_ablation_variant(self.patches, 2)
        self.assertTrue(np.all(transformed[:, 2] == 0.0))
        self.assertTrue(np.array_equal(transformed[:, :2], self.patches[:, :2]))
        self.assertTrue(np.array_equal(transformed[:, 3:], self.patches[:, 3:]))


if __name__ == "__main__":
    unittest.main()
