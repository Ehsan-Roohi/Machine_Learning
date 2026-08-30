from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "generate_cases.py"
SPEC = importlib.util.spec_from_file_location("generate_cases", MODULE_PATH)
assert SPEC and SPEC.loader
GEN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GEN
SPEC.loader.exec_module(GEN)


def test_surface_matches_original_discretization() -> None:
    for geometry in GEN.APEX_X:
        points, lines = GEN.wall_points_and_lines(geometry)
        assert len(points) == 1043
        assert len(lines) == 1040
        assert lines[0] == (1, 1, 1, 2)
        assert lines[219] == (220, 1, 220, 221)
        assert lines[220] == (221, 1, 222, 223)
        assert lines[979] == (980, 1, 981, 982)
        assert lines[980] == (981, 2, 983, 984)
        assert lines[-1] == (1040, 2, 1042, 1043)
        assert points[982] == (0.22, 0.0)
        assert points[1012] == (GEN.APEX_X[geometry], GEN.H_P)
        assert points[-1] == (0.24, 0.0)


def test_production_matrix_and_schema(tmp_path: Path) -> None:
    cases = GEN.generate(tmp_path, "production")
    assert [path.name for path in cases] == [GEN.case_id(*item) for item in GEN.CASE_MATRIX]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest) == 6
    for case in manifest:
        case_dir = tmp_path / case["case_id"]
        text = (case_dir / "in.moment_pilot").read_text()
        assert "pflux/grid" in text
        assert "eflux/grid" in text
        assert "press shx shy shz ke erot evib etot" in text
        assert case["settings"]["nblocks"] == 4
        assert len(case["column_schema"]["grid"]) == 25
        assert len(case["column_schema"]["wall"]) == 15


def test_freestream_reproduces_archived_values() -> None:
    values = GEN.freestream(6.0, 0.1, GEN.SETTINGS["production"])
    assert abs(values["stream_speed_m_per_s"] - 1936.08) < 0.1
    assert abs(values["number_density_m_minus_3"] - 4.38942e20) / 4.38942e20 < 1e-12
    assert values["timestep_s"] <= values["dt_collision_limit_s"]
    assert values["timestep_s"] <= values["dt_advection_limit_s"]
