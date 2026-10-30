from __future__ import annotations

import os
import stat

import pytest

from reos.cli import build_parser, main, open_checkpoint_database


def test_cli_requires_explicit_session_id_for_normal_runs() -> None:
    assert main(["Assess the launch plan"]) == 2


def test_cli_accepts_explicit_session_and_database() -> None:
    args = build_parser().parse_args(
        ["Assess the launch plan", "--session-id", "launch-01", "--database", "data/run.sqlite"]
    )

    assert args.request == "Assess the launch plan"
    assert args.session_id == "launch-01"
    assert args.database == "data/run.sqlite"


def test_cli_supports_doctor_without_request_or_session() -> None:
    args = build_parser().parse_args(["--doctor"])

    assert args.doctor is True
    assert args.request is None
    assert args.session_id is None


def test_cli_supports_a_provider_backed_smoke_run() -> None:
    args = build_parser().parse_args(["--live-smoke"])

    assert args.live_smoke is True
    assert args.request is None


def test_checkpoint_database_is_created_with_private_permissions(tmp_path) -> None:
    database = tmp_path / "private" / "agent.sqlite"
    connection = open_checkpoint_database(database)
    try:
        assert database.exists()
        if os.name == "posix":
            assert stat.S_IMODE(database.stat().st_mode) == 0o600
            assert stat.S_IMODE(database.parent.stat().st_mode) == 0o700
    finally:
        connection.close()
