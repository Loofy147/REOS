from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

from reos.preflight import build_preflight_report
from reos.quality import EvidenceReference


SMOKE_REQUEST = (
    "ATHENA provider smoke test. In no more than 8 concise bullet points, "
    "identify a plausible operational risk, one counterargument, and the next "
    "test that would distinguish between them. Label unverified claims as "
    "HYPOTHESIS or UNKNOWN. Do not claim external research or tool use."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reos",
        description="Run ATHENA or inspect whether its configured runtime is ready.",
    )
    parser.add_argument(
        "request",
        nargs="?",
        help="The question or decision ATHENA should assess.",
    )
    parser.add_argument(
        "--session-id",
        help="Required for normal runs. Reuse only to continue the same session; use a unique value per user/project.",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("ATHENA_DB_PATH", "athena-checkpoints.sqlite"),
        help="SQLite checkpoint database path (default: ATHENA_DB_PATH or ./athena-checkpoints.sqlite).",
    )
    parser.add_argument(
        "--evidence-file",
        type=Path,
        help="JSON evidence register. Each reference needs reference_id, source, observed_at, locator, and excerpt.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--doctor",
        "--preflight",
        action="store_true",
        help="Check local runtime, provider integrations, credentials, and checkpoint storage without making model calls.",
    )
    mode.add_argument(
        "--live-smoke",
        action="store_true",
        help="Make one low-scope, real provider-backed ATHENA run using temporary in-memory state.",
    )
    return parser


def open_checkpoint_database(database_path: Path) -> sqlite3.Connection:
    """Open a local checkpoint DB with owner-only POSIX permissions."""
    database_path = database_path.expanduser()
    database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        database_fd = os.open(str(database_path), os.O_CREAT | os.O_RDWR, 0o600)
        os.close(database_fd)
        os.chmod(database_path, 0o600)
    return sqlite3.connect(str(database_path), check_same_thread=False)


def load_evidence_references(path: Path | None) -> tuple[EvidenceReference, ...]:
    """Load explicit, user-reviewed evidence excerpts; never fetch URLs implicitly."""
    if path is None:
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read evidence register {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evidence register is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("references"), list):
        raise ValueError("Evidence register must be an object with a 'references' array.")

    references: list[EvidenceReference] = []
    required = ("reference_id", "source", "observed_at", "locator", "excerpt")
    for index, item in enumerate(payload["references"]):
        if not isinstance(item, dict):
            raise ValueError(f"Evidence reference at index {index} must be an object.")
        missing = [key for key in required if not isinstance(item.get(key), str) or not item[key].strip()]
        if missing:
            raise ValueError(
                f"Evidence reference at index {index} is missing non-empty fields: {', '.join(missing)}."
            )
        references.append(
            EvidenceReference(
                reference_id=item["reference_id"].strip(),
                source=item["source"].strip(),
                observed_at=item["observed_at"].strip(),
                locator=item["locator"].strip(),
                excerpt=item["excerpt"].strip(),
            )
        )
    return tuple(references)


def _print_preflight(report: dict[str, object]) -> None:
    checks = report["checks"]
    print("ATHENA preflight: READY" if report["ok"] else "ATHENA preflight: BLOCKED")
    for check in checks:
        print(f"[{check['status']}] {check['name']}: {check['detail']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        evidence_references = load_evidence_references(args.evidence_file)
    except ValueError as exc:
        print(f"ATHENA evidence register invalid: {exc}", file=sys.stderr)
        return 2

    analysis_spec = os.environ.get("ATHENA_ANALYSIS_MODEL", "").strip()
    review_spec = os.environ.get("ATHENA_REVIEW_MODEL", "").strip()
    preflight = build_preflight_report(
        analysis_model=analysis_spec,
        review_model=review_spec,
        database_path=args.database,
    )

    if args.doctor:
        _print_preflight(preflight)
        return 0 if preflight["ok"] else 2

    if args.live_smoke:
        request = SMOKE_REQUEST
        session_id = f"athena-smoke-{uuid.uuid4().hex}"
    else:
        if not args.request or not args.request.strip():
            print("Provide a request, or use --doctor / --live-smoke.", file=sys.stderr)
            return 2
        if not args.session_id or not args.session_id.strip():
            print("Normal ATHENA runs require an explicit --session-id.", file=sys.stderr)
            return 2
        request = args.request
        session_id = args.session_id.strip()

    if not preflight["ok"]:
        _print_preflight(preflight)
        return 2

    # Enable strict checkpoint deserialization before importing the checkpointer.
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

    try:
        from langchain.chat_models import init_chat_model
        from langgraph.checkpoint.memory import InMemorySaver
        from reos.agents.athena import AthenaAgent

        if not args.live_smoke:
            from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        print(
            "ATHENA runtime is not installed. Install REOS with its 'lang' extra "
            "and the matching provider extra (for example: pip install -e '.[lang,openai]'). "
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 2

    connection: sqlite3.Connection | None = None
    try:
        analysis_model = init_chat_model(analysis_spec, temperature=0)
        review_model = init_chat_model(review_spec, temperature=0)

        if args.live_smoke:
            checkpointer = InMemorySaver()
        else:
            connection = open_checkpoint_database(Path(args.database))
            checkpointer = SqliteSaver(connection)
            checkpointer.setup()

        agent = AthenaAgent(
            analysis_model=analysis_model,
            review_model=review_model,
            checkpointer=checkpointer,
        )
        response = agent.invoke(
            request,
            session_id=session_id,
            evidence_references=evidence_references,
        )
    except Exception as exc:  # CLI boundary: surface actionable errors without a traceback by default.
        print(f"ATHENA run failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()

    if args.live_smoke:
        print("ATHENA LIVE SMOKE: PASS — both configured providers completed the graph.")
    print("\n## ATHENA — analysis\n")
    print(response.analysis)
    print("\n## Independent challenge\n")
    print(response.critique)
    print("\n## Synthesized answer\n")
    print(response.answer)
    print(f"\n## Deterministic quality gate: {response.quality_report.status.value}")
    for finding in response.quality_report.findings:
        print(f"[{finding.severity.value}] {finding.code.value}: {finding.message}")
        if finding.excerpt:
            print(f"  Evidence: {finding.excerpt}")
    persistence = "ephemeral" if args.live_smoke else "SQLite"
    evidence_count = sum(len(turn.evidence_references) for turn in response.turn_history)
    print(
        f"\nAudit history: {len(response.turn_history)} turns | "
        f"Registered evidence references retained: {evidence_count}"
    )
    for index, turn in enumerate(response.turn_history, start=1):
        print(
            f"  Turn {index}: {turn.quality_report.status.value} | "
            f"{len(turn.evidence_references)} evidence references"
        )
    print(f"\nSession: {response.session_id} | Turns: {response.turns_count} | State: {persistence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
