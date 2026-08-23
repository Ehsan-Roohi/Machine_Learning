"""Public package interface for FlowMLLab."""

from .project import FlowMLLabProject, ProjectLayoutError

__all__ = ["FlowMLLabProject", "ProjectLayoutError", "__version__"]
__version__ = "1.0.0"
