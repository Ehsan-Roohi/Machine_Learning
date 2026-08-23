#!/usr/bin/env python3
"""Release gate for the public MIE 690A course tree."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATA_SHA256 = "09b96b744ee4d18126d8dcc92feb60e128774a1b4d41bb3d8c90a63ccfbabc36"


REQUIRED = [
    "README.md",
    "START_HERE.md",
    "COURSE_MAP.md",
    "requirements.txt",
    "CITATION.cff",
    "data/cavity_data.npz",
    "data/case_quality.csv",
    "common/w4utils.py",
    "common/w5_common.py",
    "common/mini_dsmc.py",
    "common/run_pod_deeponet_validation.py",
    "notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb",
    "notebooks/P0_Project_Setup.ipynb",
    "notebooks/P1_Re_Generalization.ipynb",
    "notebooks/P2_Physics_Guided_DNN.ipynb",
    "notebooks/P3_POD_Study.ipynb",
    "notebooks/P4_Uncertainty_Study.ipynb",
    "notebooks/P5_Rarefied_Cavity.ipynb",
    "notebooks/P6_FP_Cavity_Closure.ipynb",
    "lectures/week01_numerical_foundations.pdf",
    "lectures/week02_supervised_learning_rarefaction.pdf",
    "lectures/week03_kinetic_dsmc.pdf",
    "lectures/week04_cavity_surrogates_deeponet.pdf",
    "lectures/weeks05_06_project_guide.pdf",
    "references/README.md",
    "references/course_references.bib",
    "results/pod_deeponet/deeponet_selection.csv",
    "results/pod_deeponet/README.md",
    "results/pod_deeponet/deeponet_metrics.csv",
    "results/pod_deeponet/deeponet_ghia_metrics.csv",
    "results/pod_deeponet/deeponet_protocol_and_timing.json",
    "results/pod_deeponet/deeponet_predictions.csv",
    "results/pod_deeponet/pod_deeponet_ghia_validation.svg",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_notebook_code(source: str, label: str) -> None:
    kept = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    if cleaned.strip():
        ast.parse(cleaned, filename=label)


def validate_notebooks() -> tuple[int, int]:
    count = 0
    code_cells = 0
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook.get("nbformat") == 4, f"unexpected nbformat: {path}"
        cells = notebook.get("cells", [])
        assert cells, f"empty notebook: {path}"
        ids = [cell.get("id") for cell in cells if cell.get("id")]
        assert len(ids) == len(set(ids)), f"duplicate cell id: {path}"
        for index, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                code_cells += 1
                parse_notebook_code("".join(cell.get("source", [])), f"{path}:{index}")
        if path.parent == ROOT / "notebooks" and path.name.startswith("P"):
            words = sum(
                len("".join(cell.get("source", [])).split())
                for cell in cells
                if cell.get("cell_type") == "markdown"
            )
            minimum = 2000 if path.name.startswith("P6_") else 1000
            assert words >= minimum, f"project notebook lacks learner guidance ({words} words): {path}"
            assert any(
                "MIE690A enriched learner edition v2" in "".join(cell.get("source", []))
                for cell in cells
            ), f"missing learner-edition marker: {path}"
        count += 1
    assert count == 16, f"expected 16 notebooks, found {count}"
    return count, code_cells


def smoke_common_baseline() -> dict[str, float]:
    sys.path.insert(0, str(ROOT / "common"))
    import w5_common  # noqa: PLC0415

    data = w5_common.require_week4_files(str(ROOT / "data" / "cavity_data.npz"))
    train_re = data["Re"][data["split"].astype(str) == "train"]
    pred = w5_common.interpolate_case(data, 275, train_re)
    report = w5_common.evaluate_prediction(data, 275, pred)
    assert report
    assert all(np.isfinite(value) for value in report.values())
    assert report["relative_L2_uv"] < 0.10, report
    return {key: float(value) for key, value in report.items()}


def validate_pod_deeponet_results() -> dict[str, float]:
    result_dir = ROOT / "results" / "pod_deeponet"
    metrics = pd.read_csv(result_dir / "deeponet_metrics.csv")
    ensemble = metrics[metrics["method"] == "three-seed POD-DeepONet ensemble"].copy()
    assert sorted(ensemble["Re"].astype(int).tolist()) == [175, 275, 375]
    assert float(ensemble["relative_L2_uv"].max()) < 0.005
    assert float(ensemble["div_l2_pred"].max()) < 1.0e-12
    assert float(ensemble["wall_rms_error"].max()) == 0.0

    ghia = pd.read_csv(result_dir / "deeponet_ghia_metrics.csv")
    assert sorted(ghia["Re"].astype(int).tolist()) == [100, 400]
    ghia_delta = np.max(
        np.abs(
            ghia[["POD_DeepONet_Ghia_Eu", "POD_DeepONet_Ghia_Ev"]].to_numpy()
            - ghia[["CFD_Ghia_Eu", "CFD_Ghia_Ev"]].to_numpy()
        )
    )
    assert float(ghia_delta) < 5.0e-4

    timing = json.loads(
        (result_dir / "deeponet_protocol_and_timing.json").read_text(encoding="utf-8")
    )
    assert timing["selected_rank"] == 3
    assert timing["selected_hidden"] == [32, 32]
    assert float(timing["speedup"]) > 100.0
    assert float(timing["CFD_final_residual"]) < 1.0e-6

    predictions = pd.read_csv(result_dir / "deeponet_predictions.csv")
    field_columns = [
        "u_Re175", "v_Re175", "u_Re275", "v_Re275", "u_Re375", "v_Re375"
    ]
    assert len(predictions) == 65 * 65
    assert predictions[["iy", "ix"]].drop_duplicates().shape[0] == 65 * 65
    assert predictions[["x", "y", *field_columns]].notna().all().all()

    binary_predictions = result_dir / "deeponet_predictions.npz"
    if binary_predictions.is_file():
        with np.load(binary_predictions, allow_pickle=False) as archive:
            assert archive["u"].shape == (3, 65, 65)
            assert archive["v"].shape == (3, 65, 65)
            assert archive["seeds"].tolist() == [690, 691, 692]

    return {
        "max_blind_relative_L2_uv": float(ensemble["relative_L2_uv"].max()),
        "max_divergence_L2": float(ensemble["div_l2_pred"].max()),
        "max_Ghia_error_change": float(ghia_delta),
        "measured_speedup": float(timing["speedup"]),
    }


def validate_pdfs() -> int:
    pdfs = sorted((ROOT / "lectures").glob("*.pdf"))
    assert len(pdfs) == 5
    for path in pdfs:
        result = subprocess.run(
            ["pdfinfo", str(path)], check=True, capture_output=True, text=True
        )
        assert "Pages:" in result.stdout and "Page size:" in result.stdout
    return len(pdfs)


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    assert not missing, "missing release files: " + ", ".join(missing)
    actual = digest(ROOT / "data" / "cavity_data.npz")
    assert actual == EXPECTED_DATA_SHA256, (actual, EXPECTED_DATA_SHA256)
    notebooks, code_cells = validate_notebooks()
    metrics = smoke_common_baseline()
    deeponet_metrics = validate_pod_deeponet_results()
    pdfs = validate_pdfs()
    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    print("MIE690A_RELEASE_QA_PASS")
    print("notebooks:", notebooks, "code cells parsed:", code_cells)
    print("lecture PDFs:", pdfs)
    print("Python files parsed:", len(python_files))
    print("dataset SHA-256:", actual)
    print("Re=275 interpolation metrics:", json.dumps(metrics, sort_keys=True))
    print("POD-DeepONet release metrics:", json.dumps(deeponet_metrics, sort_keys=True))


if __name__ == "__main__":
    main()
