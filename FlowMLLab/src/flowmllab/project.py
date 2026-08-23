"""Project discovery and subprocess execution shared by the CLI front ends."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


class ProjectLayoutError(RuntimeError):
    """Raised when the executable evidence tree cannot be located or audited."""


REQUIRED_EVIDENCE = (
    "CITATION.cff",
    "LICENSE",
    "common/article_validation.py",
    "common/run_cavity_pressure_validation.py",
    "common/run_pod_deeponet_validation.py",
    "data/cavity_data.npz",
    "qa/validate_course_release.py",
    "results/article_validation/pressure_validation_protocol.json",
    "results/dsmc_validation/mohammadzadeh_fig3_dsmc_points.csv",
    "results/pod_deeponet/deeponet_protocol_and_timing.json",
)


@dataclass(frozen=True)
class FlowMLLabProject:
    """Resolved repository and evidence roots for a FlowMLLab execution."""

    software_root: Path
    evidence_root: Path

    @classmethod
    def discover(cls, root: str | Path | None = None) -> "FlowMLLabProject":
        """Find the software and retained-evidence trees without notebook state."""
        if root is not None:
            starts = [Path(root).expanduser().resolve()]
        elif os.environ.get("FLOWMLLAB_ROOT"):
            starts = [Path(os.environ["FLOWMLLAB_ROOT"]).expanduser().resolve()]
        else:
            starts = [Path.cwd().resolve(), Path(__file__).resolve()]

        candidates: list[Path] = []
        for start in starts:
            candidates.extend((start, *start.parents))
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            direct = candidate
            legacy = candidate / "MIE690A_AI_in_Fluid_Mechanics"
            for evidence in (direct, legacy):
                if (evidence / "common" / "article_validation.py").is_file():
                    software = candidate / "FlowMLLab"
                    if not (software / "pyproject.toml").is_file():
                        software = Path(__file__).resolve().parents[2]
                    return cls(software_root=software, evidence_root=evidence)
        raise ProjectLayoutError(
            "FlowMLLab evidence tree not found. Run inside the Machine_Learning "
            "checkout or pass --root/ set FLOWMLLAB_ROOT."
        )

    def missing_evidence(self, required: Iterable[str] = REQUIRED_EVIDENCE) -> list[str]:
        return [item for item in required if not (self.evidence_root / item).is_file()]

    def require_layout(self) -> None:
        missing = self.missing_evidence()
        if missing:
            raise ProjectLayoutError("missing required evidence: " + ", ".join(missing))

    def run_python(self, relative_script: str, *args: str) -> subprocess.CompletedProcess[str]:
        script = self.evidence_root / relative_script
        if not script.is_file():
            raise ProjectLayoutError(f"required program not found: {relative_script}")
        command = [sys.executable, str(script), *args]
        return subprocess.run(command, cwd=self.evidence_root, check=True, text=True)
