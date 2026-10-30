from __future__ import annotations

import pytest

from reos.cli import build_parser, open_checkpoint_database


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



def test_checkpoint_database_is_created_with_private_permissions(tmp_path) -> None:
    import os
    import stat

    database = tmp_path / "private" / "agent.sqlite"
    connection = open_checkpoint_database(database)
    try:
        assert database.exists()
        if os.name == "posix":
            assert stat.S_IMODE(database.stat().st_mode) == 0o600
            assert stat.S_IMODE(database.parent.stat().st_mode) == 0o700
    finally:
        connection.close()
