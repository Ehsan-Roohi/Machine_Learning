import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:  # The Unity production environment provides torch.
    raise unittest.SkipTest("Stage-1 V2 neural tests require PyTorch") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "stage1_v2.py"
SPEC = importlib.util.spec_from_file_location("stage1_v2", MODULE_PATH)
stage1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = stage1
SPEC.loader.exec_module(stage1)


def six_ring_table() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    cases = [f"{geom}_{i}" for geom in ("BWD", "FWD", "ISO") for i in range(2)]
    rows = []
    for case_i, case_id in enumerate(cases):
        geom = case_id.split("_")[0]
        for surface_i, s01 in enumerate(np.linspace(0, 1, 4)):
            row = {
                "case_id": case_id,
                "surface_i": surface_i,
                "geom": geom,
                "Ma": 4.0 + 2.0 * (case_i % 3),
                "Kn": 0.1 + 0.23 * (case_i % 2),
                "s01": s01,
                "Cp": rng.normal(),
                "Cq": rng.normal(),
                "tau_abs": abs(rng.normal()),
                "gate_sample_weight": 1.0,
            }
            for j in range(7):
                row[f"operator_case_f{j:03d}"] = case_i + 0.1 * j
            for j in range(5):
                row[f"operator_surface_f{j:03d}"] = s01 + 0.01 * j
            for ring in (
                "ring_0p0_0p1",
                "ring_0p1_0p25",
                "ring_0p25_0p5",
                "ring_0p5_1p0",
                "ring_1p0_2p0",
                "ring_2p0_inf",
            ):
                row[f"{ring}_u_mean"] = 10.0 * case_i + s01
                row[f"{ring}_T_std"] = case_i - s01
            rows.append(row)
    return pd.DataFrame(rows)


class Stage1V2Tests(unittest.TestCase):
    def test_matched_count_spatial_controls(self):
        frame = six_ring_table()
        schema = stage1.v1.infer_schema(frame)
        masks = stage1.mask_configs(schema)
        self.assertEqual(int(masks["M_R0p5"].sum()), 3)
        self.assertEqual(int(masks["M_far_K3"].sum()), 3)
        self.assertEqual(int(masks["M_interleaved_K3"].sum()), 3)
        self.assertTrue(np.array_equal(masks["M_R0p5"], [1, 1, 1, 0, 0, 0]))

    def test_shared_ring_operator_has_fixed_output_shape(self):
        model = stage1.NestedRingOperator(
            case_dim=7,
            surface_dim=15,
            ring_dim=2,
            n_rings=6,
            latent=12,
            hidden=16,
            depth=2,
            dropout=0.0,
            full_residual_scale=0.05,
        )
        case = torch.randn(5, 7)
        surface = torch.randn(5, 15)
        rings = torch.randn(5, 6, 2)
        empty = torch.zeros(5, 6, dtype=torch.bool)
        full = torch.ones(5, 6, dtype=torch.bool)
        pred0, attention0 = model(case, surface, rings, empty, return_attention=True)
        predf, attentionf = model(case, surface, rings, full, return_attention=True)
        self.assertEqual(tuple(pred0.shape), (5, 1))
        self.assertEqual(tuple(predf.shape), (5, 1))
        self.assertTrue(torch.allclose(attention0, torch.zeros_like(attention0)))
        self.assertTrue(torch.allclose(attentionf.sum(dim=1), torch.ones(5), atol=1e-6))

    def test_case_profile_permutation_is_coherent_and_deterministic(self):
        frame = six_ring_table()
        schema = stage1.v1.infer_schema(frame)
        prep = stage1.v1.fit_preprocessor(frame, schema)
        rings = stage1.transform_arrays(frame, schema, prep)[2]
        first, mapping1 = stage1.permute_case_profiles(frame, rings, 55)
        second, mapping2 = stage1.permute_case_profiles(frame, rings, 55)
        self.assertEqual(mapping1, mapping2)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, rings))
        for destination, source in mapping1.items():
            self.assertEqual(destination.split("_")[0], source.split("_")[0])

    def test_ensemble_is_formed_before_profile_error(self):
        rows = []
        true = np.asarray([1.0, 2.0, 3.0])
        for seed, delta in [(101, 1.0), (202, -1.0)]:
            for surface_i, value in enumerate(true):
                rows.append(
                    {
                        "scheme": "loco",
                        "outer_group": "case_a",
                        "seed": seed,
                        "target": "Cp",
                        "config": "M_R0p5",
                        "case_id": "case_a",
                        "surface_i": surface_i,
                        "s01": surface_i / 2,
                        "true": value,
                        "pred": value + delta,
                    }
                )
        _, metrics = stage1.ensemble_case_metrics(pd.DataFrame(rows), expected_seeds=2)
        self.assertAlmostEqual(float(metrics.iloc[0]["relL2"]), 0.0, places=12)

    def test_incomplete_ensemble_is_rejected(self):
        rows = []
        for surface_i in range(2):
            rows.append(
                {
                    "scheme": "loco",
                    "outer_group": "case_a",
                    "seed": 101,
                    "target": "Cp",
                    "config": "M_R0p5",
                    "case_id": "case_a",
                    "surface_i": surface_i,
                    "s01": float(surface_i),
                    "true": float(surface_i),
                    "pred": float(surface_i),
                }
            )
        with self.assertRaisesRegex(RuntimeError, "expected exactly 5 seeds"):
            stage1.ensemble_case_metrics(pd.DataFrame(rows), expected_seeds=5)

    def test_radius_selection_is_absolute_and_censor_aware(self):
        truth = np.asarray([1.0, 2.0, 3.0])
        case_ids = np.asarray(["a", "a", "a"])
        predictions = {
            "M_R0p1": truth + 1.0,
            "M_R0p25": truth + 0.20,
            "M_R0p5": truth + 0.01,
            "M_R1": truth + 0.005,
            "M_R2": truth,
            "M_full": truth,
        }
        selected, censored, table = stage1.choose_inner_radius(
            predictions,
            truth,
            case_ids,
            absolute_tolerance=0.10,
            noninferiority_margin=0.01,
        )
        self.assertEqual(selected, "M_R0p5")
        self.assertFalse(censored)
        self.assertEqual(int(table["qualified"].sum()), 4)

        bad = {name: value + 2.0 for name, value in predictions.items()}
        selected, censored, _ = stage1.choose_inner_radius(
            bad,
            truth,
            case_ids,
            absolute_tolerance=0.10,
            noninferiority_margin=0.01,
        )
        self.assertEqual(selected, "M_full")
        self.assertTrue(censored)


if __name__ == "__main__":
    unittest.main()
