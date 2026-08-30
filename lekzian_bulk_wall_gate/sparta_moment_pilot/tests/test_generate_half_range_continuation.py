from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "generate_half_range_continuation.py"
SPEC = importlib.util.spec_from_file_location("generate_half_range_continuation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_render_samples_every_timestep() -> None:
    metadata = {"case_id": "ISO_Ma6_Kn0p1", "wall_temperature_K": 300.0}
    text = MODULE.render_input(metadata, 5000, 1000)
    assert "read_restart output/final.restart" in text
    assert "reset_timestep 1" in text
    assert "dump dCollisionLong tally all 1" in text
    assert "output/half_range_long/collisions.dat" in text
    assert "run 5000" in text


def test_generate_refuses_existing_output(tmp_path: Path) -> None:
    for case_id in MODULE.CASE_IDS:
        case = tmp_path / case_id
        (case / "output").mkdir(parents=True)
        (case / "output" / "final.restart").write_bytes(b"restart")
        (case / "metadata.json").write_text(
            json.dumps({"case_id": case_id, "wall_temperature_K": 300.0})
        )
    MODULE.generate(tmp_path, 20, 10)
    try:
        MODULE.generate(tmp_path, 20, 10)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output should not be overwritten")
