"""Direct Simulation Monte Carlo reproduction entry points."""

from __future__ import annotations

from flowmllab.project import FlowMLLabProject


def reproduce(project: FlowMLLabProject) -> None:
    """Rebuild the retained DSMC wall-pressure validation figure and metrics."""
    project.run_python("common/article_validation.py", "dsmc", "--root", str(project.evidence_root))


__all__ = ["reproduce"]
