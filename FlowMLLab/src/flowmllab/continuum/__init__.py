"""Continuum-flow reproduction entry points."""

from __future__ import annotations

from flowmllab.project import FlowMLLabProject


def reproduce(project: FlowMLLabProject, *, recompute: bool = False) -> None:
    """Reproduce pressure validation, optionally regenerating both CFD runs."""
    if recompute:
        project.run_python("common/run_cavity_pressure_validation.py")
    project.run_python("common/article_validation.py", "pressure", "--root", str(project.evidence_root))


__all__ = ["reproduce"]
