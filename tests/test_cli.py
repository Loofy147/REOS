from __future__ import annotations

import pytest

from reos.cli import build_parser


def test_cli_requires_explicit_session_id() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["Assess the launch plan"])

    assert error.value.code == 2


def test_cli_accepts_explicit_session_and_database() -> None:
    args = build_parser().parse_args(
        ["Assess the launch plan", "--session-id", "launch-01", "--database", "data/run.sqlite"]
    )

    assert args.session_id == "launch-01"
    assert args.database == "data/run.sqlite"
