"""Release and evidence validation entry points."""

from __future__ import annotations

from flowmllab.project import FlowMLLabProject


def validate(project: FlowMLLabProject, *, structural_only: bool = False) -> None:
    """Check the package/evidence contract and optionally run the full release QA."""
    project.require_layout()
    if not structural_only:
        project.run_python("qa/validate_course_release.py")


__all__ = ["validate"]
