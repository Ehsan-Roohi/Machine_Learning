from __future__ import annotations

import unittest

from flowmllab.cli import build_parser


class CommandLineTests(unittest.TestCase):
    def test_parser_accepts_public_commands(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["validate", "--structural-only"]).command, "validate")
        self.assertEqual(parser.parse_args(["reproduce", "continuum"]).study, "continuum")
        self.assertEqual(parser.parse_args(["reproduce", "pod-deeponet"]).study, "pod-deeponet")
        self.assertEqual(parser.parse_args(["reproduce", "dsmc"]).study, "dsmc")


if __name__ == "__main__":
    unittest.main()
