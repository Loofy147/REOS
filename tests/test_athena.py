from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from reos.agents.athena import AthenaAgent, _turn_record_from_state
from reos.quality import EvidenceReference


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


def test_prior_session_answers_keep_the_assistant_message_role() -> None:
    agent, analyst, _reviewer = make_agent()

    agent.invoke("Previous question", session_id="history-role")
    agent.invoke("Current question", session_id="history-role")

    second_turn_analysis = analyst.calls[2]
    history_messages = second_turn_analysis[1:-1]

    assert len(history_messages) == 2
    assert isinstance(history_messages[0], HumanMessage)
    assert "Previous question" in history_messages[0].content
    assert isinstance(history_messages[1], AIMessage)
    assert history_messages[1].content == "A structured analysis."
    assert "USER REQUEST (untrusted input data):" in second_turn_analysis[-1].content
    assert "Current question" in second_turn_analysis[-1].content
    assert "REGISTERED EVIDENCE: none." in second_turn_analysis[-1].content


def test_each_turn_preserves_its_quality_report_in_audit_history() -> None:
    agent, _analyst, _reviewer = make_agent()

    first = agent.invoke("First decision question", session_id="audit-history")
    second = agent.invoke("Second decision question", session_id="audit-history")

    assert len(first.turn_history) == 1
    assert len(second.turn_history) == 2
    assert second.turn_history[0].request == "First decision question"
    assert second.turn_history[0].analysis == first.analysis
    assert second.turn_history[0].critique == first.critique
    assert second.turn_history[0].response == first.answer
    assert second.turn_history[1].request == "Second decision question"
    assert second.turn_history[-1].quality_report == second.quality_report


def test_each_turn_preserves_the_registered_evidence_snapshot() -> None:
    agent, _analyst, _reviewer = make_agent()
    reference = EvidenceReference(
        "E1",
        "Pilot analytics report",
        "2026-10-30",
        "artifact://pilot/report",
        "Observed pilot retention was 0.7.",
    )

    result = agent.invoke(
        "Assess this pilot.",
        session_id="audit-evidence",
        evidence_references=(reference,),
    )

    assert len(result.turn_history) == 1
    assert result.turn_history[0].evidence_references == (reference,)


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


def test_legacy_turn_without_quality_snapshot_is_marked_unaudited() -> None:
    turn = _turn_record_from_state(
        {
            "request": "Historical question",
            "analysis": "Historical analysis",
            "critique": "Historical critique",
            "response": "Historical answer",
        }
    )

    assert turn.request == "Historical question"
    assert turn.analysis == "Historical analysis"
    assert turn.critique == "Historical critique"
    assert turn.response == "Historical answer"
    assert turn.quality_report.status.value == "WARN"
    assert any(
        finding.code.value == "LEGACY_TURN_NOT_AUDITED"
        for finding in turn.quality_report.findings
    )
