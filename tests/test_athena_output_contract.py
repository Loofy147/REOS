from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from reos.agents.athena import AthenaAgent, AthenaOutputError


class SequenceChatModel:
    """Deterministic model double with one configured output per invocation."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def invoke(self, messages: list[Any], config: Any = None) -> AIMessage:
        return AIMessage(content=self.responses.pop(0))


@pytest.mark.parametrize("empty_stage", ["analysis", "review", "synthesis"])
def test_empty_model_output_fails_and_failed_turn_is_not_recorded(empty_stage: str) -> None:
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

    # Recovering the same thread must count only the successful turn.
    analyst.responses = ["recovered analysis", "recovered synthesis"]
    reviewer.responses = ["recovered challenge"]
    recovered = agent.invoke("Retry the same session", session_id="empty-output")
    assert recovered.turns_count == 1


def test_sensitive_recommendation_is_blocked_before_the_turn_is_recorded() -> None:
    analyst = SequenceChatModel(
        ["Operational data is needed.", "Require customers to upload raw sales records."]
    )
    reviewer = SequenceChatModel(["The raw-data request is unnecessary and risky."])
    agent = AthenaAgent(
        analysis_model=analyst,
        review_model=reviewer,
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(AthenaOutputError, match="quality gate"):
        agent.invoke("Assess a business pilot.", session_id="privacy-gate")

    # A retry of the same thread must still be the first completed turn.
    analyst.responses = [
        "Operational evidence can be tested safely.",
        "Decision: run a limited pilot. Falsification test: stop if users do not complete the core workflow.",
    ]
    reviewer.responses = ["Use minimized and aggregated data only."]
    recovered = agent.invoke("Retry with a privacy-safe plan.", session_id="privacy-gate")

    assert recovered.turns_count == 1


def test_unsupported_quantitative_claim_is_exposed_in_quality_report() -> None:
    analyst = SequenceChatModel(
        ["The campaign will convert 30% of visitors.", "Decision: defer launch. Falsification test: measure paid conversion before reconsidering."]
    )
    reviewer = SequenceChatModel(["The conversion estimate has no source or observed denominator."])
    agent = AthenaAgent(
        analysis_model=analyst,
        review_model=reviewer,
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke("We have no conversion data.", session_id="numeric-gate")

    assert result.quality_report.status.value == "WARN"
    assert any(
        finding.code.value == "UNSUPPORTED_NUMERIC_CLAIM"
        for finding in result.quality_report.findings
    )
