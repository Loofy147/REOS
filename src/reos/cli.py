from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reos",
        description="Run an independent ATHENA analysis and challenge cycle.",
    )
    parser.add_argument("request", help="The question or decision ATHENA should assess.")
    parser.add_argument(
        "--session-id",
        required=True,
        help="Required memory namespace. Reuse only to continue the same session; use a unique value per user/project.",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("ATHENA_DB_PATH", "athena-checkpoints.sqlite"),
        help="SQLite checkpoint database path (default: ATHENA_DB_PATH or ./athena-checkpoints.sqlite).",
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    analysis_spec = os.environ.get("ATHENA_ANALYSIS_MODEL", "").strip()
    review_spec = os.environ.get("ATHENA_REVIEW_MODEL", "").strip()
    if not analysis_spec or not review_spec:
        print(
            "Configure ATHENA_ANALYSIS_MODEL and ATHENA_REVIEW_MODEL with "
            "provider-prefixed LangChain model identifiers.",
            file=sys.stderr,
        )
        return 2
    if analysis_spec == review_spec:
        print(
            "ATHENA_ANALYSIS_MODEL and ATHENA_REVIEW_MODEL must identify distinct models.",
            file=sys.stderr,
        )
        return 2

    # Enable strict checkpoint deserialization before importing the checkpointer.
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

    try:
        from langchain.chat_models import init_chat_model
        from langgraph.checkpoint.sqlite import SqliteSaver
        from reos.agents.athena import AthenaAgent
    except ImportError as exc:
        print(
            "ATHENA runtime is not installed. Install REOS with its 'lang' extra "
            "and the matching provider extra (for example: pip install -e '.[lang,openai]'). "
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        analysis_model = init_chat_model(analysis_spec, temperature=0)
        review_model = init_chat_model(review_spec, temperature=0)
        database_path = Path(args.database)
        connection = open_checkpoint_database(database_path)
        try:
            checkpointer = SqliteSaver(connection)
            checkpointer.setup()
            agent = AthenaAgent(
                analysis_model=analysis_model,
                review_model=review_model,
                checkpointer=checkpointer,
            )
            response = agent.invoke(args.request, session_id=args.session_id)
        finally:
            connection.close()
    except Exception as exc:  # CLI boundary: surface actionable errors without a traceback by default.
        print(f"ATHENA run failed: {exc}", file=sys.stderr)
        return 1

    print("\n## ATHENA — analysis\n")
    print(response.analysis)
    print("\n## Independent challenge\n")
    print(response.critique)
    print("\n## Synthesized answer\n")
    print(response.answer)
    print(f"\nSession: {response.session_id} | Turns: {response.turns_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
