from __future__ import annotations

import os
import stat

import pytest

from reos.cli import build_parser, load_evidence_references, main, open_checkpoint_database


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


def test_cli_accepts_an_evidence_register_path() -> None:
    args = build_parser().parse_args(
        ["Assess the launch", "--session-id", "launch-02", "--evidence-file", "evidence.json"]
    )

    assert str(args.evidence_file) == "evidence.json"


def test_cli_loads_retrieved_evidence_excerpts(tmp_path) -> None:
    evidence_file = tmp_path / "evidence.json"
    evidence_file.write_text(
        """{
          "references": [{
            "reference_id": "E1",
            "source": "Pilot analytics export",
            "observed_at": "2026-10-30",
            "locator": "artifact://pilot/metrics",
            "excerpt": "Observed conversion rate was 5% for this pilot cohort."
          }]
        }""",
        encoding="utf-8",
    )

    references = load_evidence_references(evidence_file)

    assert len(references) == 1
    assert references[0].reference_id == "E1"
    assert "5%" in references[0].excerpt


def test_cli_rejects_evidence_register_without_source_excerpt(tmp_path) -> None:
    evidence_file = tmp_path / "evidence.json"
    evidence_file.write_text(
        """{
          "references": [{
            "reference_id": "E1",
            "source": "Pilot analytics export",
            "observed_at": "2026-10-30",
            "locator": "artifact://pilot/metrics"
          }]
        }""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="excerpt"):
        load_evidence_references(evidence_file)
