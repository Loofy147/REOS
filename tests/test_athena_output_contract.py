from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from reos.agents.athena import AthenaAgent


class StaticChatModel:
    """Small deterministic model double; no network calls or credentials."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any], config: dict[str, Any] | None = None) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.response)


def message_text(messages: list[Any]) -> str:
    return "\n".join(str(message.content) for message in messages)


def make_agent() -> tuple[AthenaAgent, StaticChatModel, StaticChatModel]:
    analyst = StaticChatModel("A structured analysis.")
    reviewer = StaticChatModel("A skeptical counter-analysis.")
    agent = AthenaAgent(
        analysis_model=analyst,
        review_model=reviewer,
        checkpointer=InMemorySaver(),
    )
    return agent, analyst, reviewer


def test_athena_runs_analysis_independent_review_and_synthesis() -> None:
    agent, analyst, reviewer = make_agent()

    result = agent.invoke(
        "Evaluate whether we should launch the first release.",
        session_id="project-review-001",
    )

    assert result.session_id == "project-review-001"
    assert result.analysis == "A structured analysis."
    assert result.critique == "A skeptical counter-analysis."
    assert result.answer
    assert result.turns_count == 1
    assert len(analyst.calls) == 2  # analysis + synthesis
    assert len(reviewer.calls) == 1
    assert "independent" in message_text(analyst.calls[0]).lower()
    assert "challenge" in message_text(reviewer.calls[0]).lower()
    assert "A skeptical counter-analysis." in message_text(analyst.calls[1])


def test_memory_isolated_between_session_ids() -> None:
    agent, analyst, _reviewer = make_agent()

    agent.invoke("PRIVATE alpha context", session_id="alpha")
    beta_result = agent.invoke("A separate beta question", session_id="beta")
    alpha_result = agent.invoke("Continue alpha", session_id="alpha")

    assert beta_result.turns_count == 1
    assert alpha_result.turns_count == 2
    assert "PRIVATE alpha context" in message_text(analyst.calls[4])
    assert "PRIVATE alpha context" not in message_text(analyst.calls[2])


@pytest.mark.parametrize(
    ("request_text", "session_id"),
    [
        ("", "valid-session"),
        ("   ", "valid-session"),
        ("valid request", ""),
        ("valid request", "   "),
    ],
)
def test_rejects_empty_request_or_session_id(request_text: str, session_id: str) -> None:
    agent, _analyst, _reviewer = make_agent()

    with pytest.raises(ValueError):
        agent.invoke(request_text, session_id=session_id)


def test_requires_distinct_model_instances_for_analysis_and_review() -> None:
    shared_model = StaticChatModel("one model")

    with pytest.raises(ValueError, match="distinct"):
        AthenaAgent(
            analysis_model=shared_model,
            review_model=shared_model,
            checkpointer=InMemorySaver(),
        )



class SequenceChatModel:
    """Deterministic model double with a distinct output for each invocation."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any], config: dict[str, Any] | None = None) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.responses.pop(0))


@pytest.mark.parametrize("empty_stage", ["analysis", "review", "synthesis"])
def test_empty_model_output_fails_the_turn_without_recording_it(empty_stage: str) -> None:
    from reos.agents.athena import AthenaOutputError

    analysis_responses = ["valid analysis", "valid synthesis"]
    review_responses = ["valid challenge"]
    if empty_stage == "analysis":
        analysis_responses[0] = ""
    elif empty_stage == "review":
        review_responses[0] = ""
    else:
        analysis_responses[1] = ""

    analyst = SequenceChatModel(analysis_responses)
    reviewer = SequenceChatModel(review_responses)
    agent = AthenaAgent(
        analysis_model=analyst,
        review_model=reviewer,
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(AthenaOutputError, match=empty_stage):
        agent.invoke("Test non-empty stage outputs", session_id="empty-output")

    # A failed analysis/review/synthesis must not increment the completed-turn ledger.
    analyst.responses = ["recovered analysis", "recovered synthesis"]
    reviewer.responses = ["recovered challenge"]
    recovered = agent.invoke("Retry the same session", session_id="empty-output")
    assert recovered.turns_count == 1
