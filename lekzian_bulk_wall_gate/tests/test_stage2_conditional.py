import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:  # Unity production environment provides PyTorch.
    raise unittest.SkipTest("Conditional Stage-2 neural tests require PyTorch") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "stage2_conditional.py"
SPEC = importlib.util.spec_from_file_location("stage2_conditional", MODULE_PATH)
stage2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = stage2
SPEC.loader.exec_module(stage2)


RINGS = (
    "ring_0p0_0p1",
    "ring_0p1_0p25",
    "ring_0p25_0p5",
    "ring_0p5_1p0",
    "ring_1p0_2p0",
    "ring_2p0_inf",
)
FLOW_SUFFIXES = tuple(
    f"{quantity}_{statistic}"
    for quantity in ("T", "logP", "u", "v")
    for statistic in ("grad_n", "grad_t", "max", "mean", "min", "nearest", "std")
)
STRUCTURAL_SUFFIXES = ("mean_r_hs", "mean_rn", "mean_rt", "npts", "std_r_hs")


def six_ring_table() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = []
    for geom in ("BWD", "FWD", "ISO"):
        for case_i in range(2):
            case_id = f"{geom}_{case_i}"
            for surface_i, s01 in enumerate(np.linspace(0, 1, 8)):
                row = {
                    "case_id": case_id,
                    "surface_i": surface_i,
                    "geom": geom,
                    "Ma": 4.0 + 2.0 * case_i,
                    "Kn": 0.1 + 0.23 * case_i,
                    "s01": s01,
                    "Cp": rng.normal(),
                    "Cq": rng.normal(),
                    "tau_abs": abs(rng.normal()),
                    "gate_sample_weight": 1.0,
                }
                for j in range(7):
                    row[f"operator_case_f{j:03d}"] = case_i + 0.1 * j
                for j in range(26):
                    row[f"operator_surface_f{j:03d}"] = s01 + 0.01 * j
                for ring_i, ring in enumerate(RINGS):
                    for suffix in FLOW_SUFFIXES:
                        row[f"{ring}_{suffix}"] = (
                            0.5 * case_i + 0.2 * s01 + 0.03 * ring_i + rng.normal(0, 0.01)
                        )
                    for suffix in STRUCTURAL_SUFFIXES:
                        row[f"{ring}_{suffix}"] = float(ring_i + 1)
                rows.append(row)
    return pd.DataFrame(rows)


class ConditionalStage2Tests(unittest.TestCase):
    def test_nonphysical_channels_are_excluded(self):
        frame = six_ring_table()
        schema = stage2.v1.infer_schema(frame)
        keep, suffixes = stage2.physical_suffix_indices(schema)
        self.assertEqual(len(keep), 28)
        self.assertEqual(set(suffixes), set(FLOW_SUFFIXES))
        self.assertTrue(set(suffixes).isdisjoint(stage2.NONPHYSICAL_SUFFIXES))

    def test_ridge_residualizer_removes_linear_base_signal(self):
        rng = np.random.default_rng(9)
        case = rng.normal(size=(100, 7)).astype(np.float32)
        surface = rng.normal(size=(100, 10)).astype(np.float32)
        x = stage2.design_matrix(case, surface)
        coefficient = rng.normal(size=(x.shape[1], 6 * 4))
        rings = (x @ coefficient).reshape(100, 6, 4).astype(np.float32)
        residualizer = stage2.ConditionalResidualizer.fit(
            case, surface, rings, ridge=1e-8
        )
        residual = residualizer.transform(case, surface, rings)
        self.assertLess(float(np.abs(residual).max()), 1e-3)

    def test_conditional_contrasts_have_exact_outer_zeros(self):
        rng = np.random.default_rng(12)
        conditional = rng.normal(size=(5, 6, 3)).astype(np.float32)
        tensor = stage2.conditional_contrast_tensor(conditional)
        self.assertEqual(tuple(tensor.shape), (5, 6, 9))
        self.assertTrue(np.allclose(tensor[:, -1, 3:6], 0.0))
        self.assertTrue(np.allclose(tensor[:, -1, 6:9], 0.0))
        self.assertTrue(
            np.allclose(tensor[:, 0, 6:9], conditional[:, 0] - conditional[:, 1])
        )

    def test_surface_shift_is_coherent_and_nonidentity(self):
        frame = six_ring_table()
        rings = np.arange(len(frame) * 6 * 2).reshape(len(frame), 6, 2).astype(np.float32)
        shifted = stage2.shifted_ring_tensor(frame, rings, 0.25)
        self.assertFalse(np.array_equal(shifted, rings))
        for case_id, block in frame.groupby("case_id"):
            loc = block.index.to_numpy()
            self.assertEqual(set(shifted[loc].reshape(-1)), set(rings[loc].reshape(-1)))

    def test_capacity_matched_correction_shapes_and_attention(self):
        model = stage2.ConditionalCorrection(
            context_dim=12,
            ring_dim=15,
            n_rings=6,
            latent=12,
            hidden=16,
            depth=2,
            dropout=0.0,
        )
        context = torch.randn(7, 12)
        rings = torch.randn(7, 6, 15)
        empty = torch.zeros(7, 6, dtype=torch.bool)
        near = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool).repeat(7, 1)
        c0, a0 = model(context, rings, empty, return_attention=True)
        cn, an = model(context, rings, near, return_attention=True)
        self.assertEqual(tuple(c0.shape), (7, 1))
        self.assertEqual(tuple(cn.shape), (7, 1))
        self.assertTrue(torch.allclose(a0, torch.zeros_like(a0)))
        self.assertTrue(torch.allclose(an.sum(dim=1), torch.ones(7), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
