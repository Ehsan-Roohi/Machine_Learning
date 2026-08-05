import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "gate_test.py"
SPEC = importlib.util.spec_from_file_location("gate_test", MODULE_PATH)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def synthetic_table(n=24):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "case_id": [f"case_{i // 4}" for i in range(n)],
            "surface_i": np.arange(n),
            "geom": ["ISO", "FWD", "BWD"] * (n // 3),
            "Ma": np.resize([4.0, 6.0, 8.0], n),
            "Kn": np.resize([0.1, 0.33], n),
            "s01": np.linspace(0, 1, n),
            "Cp": rng.normal(size=n),
            "Cq": rng.normal(size=n),
            "tau_abs": rng.normal(size=n),
            "gate_sample_weight": np.ones(n),
        }
    )
    for j in range(7):
        frame[f"operator_case_f{j:03d}"] = np.resize(np.arange(6, dtype=float), n) + 0.01 * j
    for j in range(4):
        frame[f"operator_surface_f{j:03d}"] = rng.normal(size=n)
    for ring in ["ring_0p0_0p1", "ring_0p1_0p25", "ring_0p25_inf"]:
        frame[f"{ring}_u_mean"] = rng.normal(size=n)
        frame[f"{ring}_T_std"] = rng.normal(size=n)
    return frame


class GateSchemaTests(unittest.TestCase):
    def test_fixed_input_dimension_for_every_configuration(self):
        df = synthetic_table()
        schema = gate.infer_schema(df)
        prep = gate.fit_preprocessor(df.iloc[:16], schema)
        configs = gate.build_configs(schema, [0.1, 0.25])
        dims = {
            gate.transform_features(df.iloc[16:], schema, prep, config, 99).shape[1]
            for config in configs
        }
        self.assertEqual(dims, {schema.input_dimension})

    def test_outer_annuli_are_masked(self):
        df = synthetic_table()
        schema = gate.infer_schema(df)
        prep = gate.fit_preprocessor(df.iloc[:16], schema)
        x0 = gate.transform_features(df.iloc[16:], schema, prep, "M0")
        start = len(schema.base_columns)
        stop = start + len(schema.bulk_columns)
        self.assertTrue(np.allclose(x0[:, start:stop], 0.0))
        self.assertTrue(np.allclose(x0[:, stop:], 0.0))

    def test_shuffle_is_deterministic_and_breaks_alignment(self):
        df = synthetic_table()
        schema = gate.infer_schema(df)
        prep = gate.fit_preprocessor(df.iloc[:16], schema)
        real = gate.transform_features(df.iloc[16:], schema, prep, "M_full")
        shuffled1 = gate.transform_features(df.iloc[16:], schema, prep, "M_shuffled", 42)
        shuffled2 = gate.transform_features(df.iloc[16:], schema, prep, "M_shuffled", 42)
        self.assertTrue(np.array_equal(shuffled1, shuffled2))
        self.assertFalse(np.array_equal(real, shuffled1))


if __name__ == "__main__":
    unittest.main()
