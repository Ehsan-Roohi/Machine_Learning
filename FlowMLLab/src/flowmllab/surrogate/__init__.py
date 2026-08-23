"""Reduced-order and operator-learning reproduction entry points."""

from __future__ import annotations

from flowmllab.project import FlowMLLabProject


def reproduce_pod_deeponet(project: FlowMLLabProject) -> None:
    """Execute model selection, three-seed training, and blind evaluation."""
    project.run_python("common/run_pod_deeponet_validation.py")


__all__ = ["reproduce_pod_deeponet"]
