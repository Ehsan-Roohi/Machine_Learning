import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Stage-3 neural tests require PyTorch") from exc


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


prepare = load("prepare_stage3_spatial")
stage3 = load("stage3_spatial")


class SpatialStage3Tests(unittest.TestCase):
    def test_target_flux_scaling_roundtrip(self):
        rng = np.random.default_rng(4)
        targets = rng.normal(size=(12, 3))
        ma = np.linspace(4, 10, 12)
        exponents = np.asarray((2.0, 3.0, 2.0))
        scaled = stage3.scale_targets(targets, ma, exponents)
        recovered = stage3.unscale_targets(scaled, ma, exponents)
        self.assertTrue(np.allclose(targets, recovered, atol=1e-12))

    def test_linear_field_patch_interpolation(self):
        x = np.linspace(-2, 2, 41)
        y = np.linspace(0, 2, 21)
        xx, yy = np.meshgrid(x, y)
        gas_xy = np.column_stack((xx.ravel(), yy.ravel()))
        gas_values = np.column_stack((
            gas_xy[:, 0], gas_xy[:, 1], gas_xy[:, 0] + gas_xy[:, 1],
            2 * gas_xy[:, 0] - gas_xy[:, 1],
        ))
        wall_xy = np.asarray([[0.0, 0.0]])
        tangent = np.asarray([[1.0, 0.0]])
        normal = np.asarray([[0.0, 1.0]])
        t_grid = np.asarray((-1.0, 0.0, 1.0))
        n_grid = np.asarray((0.1, 0.5, 1.0))
        patch = prepare.sample_wall_patches(
            gas_xy, gas_values, wall_xy, tangent, normal, 1.0,
            t_grid, n_grid, neighbours=1, valid_distance_hs=0.2,
        )
        self.assertEqual(patch.shape, (1, 6, 3, 3))
        self.assertTrue(np.allclose(patch[0, 0], np.tile(t_grid, (3, 1)), atol=1e-6))
        self.assertTrue(np.allclose(patch[0, 1], np.tile(n_grid[:, None], (1, 3)), atol=1e-6))

    def test_masks_are_matched_and_directional(self):
        t_grid = np.linspace(-3, 3, 21)
        n_grid = np.linspace(0.05, 3, 11)
        tangent = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        normal = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        masks = stage3.make_spatial_masks(t_grid, n_grid, tangent, normal, 0.75)
        near_count = masks["P_near"].sum(axis=(1, 2))
        far_count = masks["P_far"].sum(axis=(1, 2))
        self.assertTrue(np.array_equal(near_count, far_count))
        self.assertTrue(np.all((masks["P_near"] & masks["P_far"]) == 0))
        self.assertTrue(np.all(masks["P0"].sum(axis=(1, 2)) == 0))
        self.assertTrue(np.all(masks["P_full"].sum(axis=(1, 2)) == 231))
        self.assertFalse(np.array_equal(masks["P_upstream"][0], masks["P_upstream"][1]))

    def test_model_is_compact_and_context_only_is_patch_invariant(self):
        model = stage3.SpatialOperator(context_dim=33, patch_channels=6)
        # Prediction always runs in eval mode.  Keeping the model in its
        # default train mode would resample Dropout in the context branch on
        # each call and falsely attribute that stochastic difference to the
        # masked patch.
        model.eval()
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertLess(count, 100_000)
        context = torch.randn(5, 33)
        patch_a = torch.randn(5, 6, 11, 21)
        patch_b = torch.randn(5, 6, 11, 21)
        empty = torch.zeros(5, 11, 21, dtype=torch.bool)
        t = torch.randn(1, 1, 11, 21)
        n = torch.randn(1, 1, 11, 21)
        pred_a = model(context, patch_a, empty, t, n)
        pred_b = model(context, patch_b, empty, t, n)
        self.assertEqual(tuple(pred_a.shape), (5, 3))
        self.assertTrue(torch.allclose(pred_a, pred_b, atol=1e-7))

    def test_surface_shift_preserves_each_case_patch_multiset(self):
        n = 12
        data = stage3.SpatialData(
            patches=np.arange(n * 6 * 2 * 2).reshape(n, 6, 2, 2).astype(np.float32),
            context=np.zeros((n, 3), np.float32), targets=np.zeros((n, 3), np.float32),
            weights=np.ones(n, np.float32),
            case_id=np.asarray(["a"] * 6 + ["b"] * 6),
            surface_i=np.tile(np.arange(6), 2), s01=np.tile(np.linspace(0, 1, 6), 2),
            Ma=np.ones(n), Kn=np.ones(n), geom=np.asarray(["g"] * n),
            tangent=np.zeros((n, 2)), normal=np.zeros((n, 2)),
            t_grid=np.asarray((0, 1)), n_grid=np.asarray((0, 1)),
            target_scale_exponents=np.asarray((2, 3, 2)),
        )
        shifted = stage3.shifted_patch_tensor(data, data.patches, 0.25)
        self.assertFalse(np.array_equal(shifted, data.patches))
        for case_id in ("a", "b"):
            loc = np.flatnonzero(data.case_id == case_id)
            self.assertEqual(set(shifted[loc].ravel()), set(data.patches[loc].ravel()))


if __name__ == "__main__":
    unittest.main()
