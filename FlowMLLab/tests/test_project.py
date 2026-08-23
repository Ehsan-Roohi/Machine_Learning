from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from flowmllab.project import FlowMLLabProject, REQUIRED_EVIDENCE


class ProjectContractTests(unittest.TestCase):
    def test_required_evidence_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for relative in REQUIRED_EVIDENCE:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            project = FlowMLLabProject(software_root=root, evidence_root=root)
            self.assertEqual(project.missing_evidence(), [])
            project.require_layout()


if __name__ == "__main__":
    unittest.main()
