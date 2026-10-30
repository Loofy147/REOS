from __future__ import annotations

import sqlite3
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from reos.agents.athena import AthenaAgent


class FixedModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any], config: Any = None) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.response)


def flatten(messages: list[Any]) -> str:
    return "\n".join(str(message.content) for message in messages)


def build_agent(checkpointer: SqliteSaver) -> tuple[AthenaAgent, FixedModel]:
    analyst = FixedModel("analysis")
    reviewer = FixedModel("challenge")
    return (
        AthenaAgent(
            analysis_model=analyst,
            review_model=reviewer,
            checkpointer=checkpointer,
        ),
        analyst,
    )


def test_sqlite_checkpointer_restores_session_after_reopening_database(tmp_path) -> None:
    database = tmp_path / "athena.sqlite"

    first_connection = sqlite3.connect(database, check_same_thread=False)
    first_saver = SqliteSaver(first_connection)
    first_saver.setup()
    first_agent, _first_analyst = build_agent(first_saver)
    first_agent.invoke("Private alpha project context", session_id="alpha")
    first_connection.close()

    second_connection = sqlite3.connect(database, check_same_thread=False)
    try:
        second_saver = SqliteSaver(second_connection)
        second_saver.setup()
        second_agent, second_analyst = build_agent(second_saver)

        continued = second_agent.invoke("Continue alpha", session_id="alpha")
        separate = second_agent.invoke("New beta session", session_id="beta")

        assert continued.turns_count == 2
        assert separate.turns_count == 1
        assert "Private alpha project context" in flatten(second_analyst.calls[0])
        assert "Private alpha project context" not in flatten(second_analyst.calls[2])
    finally:
        second_connection.close()
