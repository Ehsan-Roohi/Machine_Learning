"""Command-line interface for reproducible FlowMLLab workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from flowmllab import __version__
from flowmllab.continuum import reproduce as reproduce_continuum
from flowmllab.kinetic import reproduce as reproduce_dsmc
from flowmllab.project import FlowMLLabProject
from flowmllab.surrogate import reproduce_pod_deeponet
from flowmllab.validation import validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowmllab",
        description="Reproduce and audit CFD-to-scientific-ML experiments.",
    )
    parser.add_argument("--version", action="version", version=f"FlowMLLab {__version__}")
    parser.add_argument("--root", type=Path, help="Machine_Learning checkout or evidence root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate release and evidence contracts")
    validate_parser.add_argument(
        "--structural-only", action="store_true", help="check required evidence paths without numerical QA"
    )

    reproduce_parser = subparsers.add_parser("reproduce", help="reproduce one retained study")
    reproduce_parser.add_argument(
        "study", choices=("continuum", "pod-deeponet", "dsmc"), help="study to execute"
    )
    reproduce_parser.add_argument(
        "--recompute", action="store_true", help="regenerate retained continuum CFD solutions before plotting"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = FlowMLLabProject.discover(args.root)
    if args.command == "validate":
        validate(project, structural_only=args.structural_only)
        print("FLOWMLLAB_VALIDATION_PASS")
        return 0
    if args.study == "continuum":
        reproduce_continuum(project, recompute=args.recompute)
    elif args.study == "pod-deeponet":
        reproduce_pod_deeponet(project)
    else:
        reproduce_dsmc(project)
    print(f"FLOWMLLAB_REPRODUCE_PASS study={args.study}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
