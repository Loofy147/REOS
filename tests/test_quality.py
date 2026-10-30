from __future__ import annotations

import json
from pathlib import Path

import pytest

from reos.quality import EvidenceReference, FindingCode, QualityStatus, audit_output


def codes(report) -> set[str]:
    return {finding.code.value for finding in report.findings}


def test_unprovided_percentage_is_flagged_as_unsupported_numeric_claim() -> None:
    report = audit_output(
        request="The budget is limited.",
        analysis="The budget is limited.",
        critique="No additional evidence.",
        answer="This approach will save 80% of the budget.",
    )

    assert report.status is QualityStatus.WARN
    assert FindingCode.UNSUPPORTED_NUMERIC_CLAIM.value in codes(report)


def test_numeric_value_supplied_by_user_is_not_flagged() -> None:
    report = audit_output(
        request="The prototype can be built in 14 days with 5 pilot customers.",
        analysis="The schedule is 14 days and the pilot has 5 customers.",
        critique="The sample remains small.",
        answer="Run the 14-day prototype with the 5 pilot customers.",
    )

    assert FindingCode.UNSUPPORTED_NUMERIC_CLAIM.value not in codes(report)


def test_proposed_threshold_requires_explicit_label_and_rationale() -> None:
    report = audit_output(
        request="There are 5 pilot customers.",
        analysis="Five customers are available.",
        critique="The sample is too small for generalization.",
        answer=(
            "PROPOSED_THRESHOLD: 2 of 5 customers. "
            "Rationale: use this only as an initial decision rule to trigger "
            "a review, not as a market estimate or universal success rate."
        ),
    )

    assert FindingCode.UNSUPPORTED_NUMERIC_CLAIM.value not in codes(report)
    assert FindingCode.PROPOSED_THRESHOLD.value in codes(report)


def test_unexplained_proposed_threshold_is_flagged() -> None:
    report = audit_output(
        request="There are 5 pilot customers.",
        analysis="Five customers are available.",
        critique="No payment evidence yet.",
        answer="The project passes if 2 of 5 customers pay.",
    )

    assert FindingCode.UNSUPPORTED_NUMERIC_CLAIM.value in codes(report)


def test_established_claim_requires_an_evidence_reference() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The market is unknown.",
        critique="The claim is unsupported.",
        answer="ESTABLISHED: Algerian customers mostly pay in cash.",
    )

    assert FindingCode.UNSOURCED_ESTABLISHED_CLAIM.value in codes(report)


def test_established_claim_with_registered_evidence_reference_is_not_flagged() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The market is unknown.",
        critique="The evidence is limited.",
        answer="ESTABLISHED: The cited regulation applies [E1].",
        evidence_references=(
            EvidenceReference("E1", "Official source", "2026-10-30", "https://example.org/source", "The cited regulation applies."),
        ),
    )

    assert FindingCode.UNSOURCED_ESTABLISHED_CLAIM.value not in codes(report)


def test_evidence_reference_requires_a_locator() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The market is unknown.",
        critique="The evidence is limited.",
        answer="ESTABLISHED: The cited regulation applies [E1].",
        evidence_references=(
            EvidenceReference("E1", "Official source", "2026-10-30", "", "The cited regulation applies."),
        ),
    )

    assert FindingCode.UNSOURCED_ESTABLISHED_CLAIM.value in codes(report)


def test_unregistered_citation_token_does_not_prove_established_claim() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The market is unknown.",
        critique="The evidence is limited.",
        answer="ESTABLISHED: The cited regulation applies [E1].",
    )

    assert FindingCode.UNSOURCED_ESTABLISHED_CLAIM.value in codes(report)


def test_sensitive_raw_business_data_request_blocks_the_output() -> None:
    report = audit_output(
        request="Assess a business pilot.",
        analysis="The product needs operational evidence.",
        critique="Privacy risks matter.",
        answer="Require customers to upload raw sales records and supplier lists.",
    )

    assert report.status is QualityStatus.BLOCK
    assert FindingCode.SENSITIVE_DATA_REQUEST.value in codes(report)


def test_redacted_or_aggregated_data_is_not_blocked() -> None:
    report = audit_output(
        request="Assess a business pilot.",
        analysis="The product needs operational evidence.",
        critique="Privacy risks matter.",
        answer="Use aggregated, anonymized sales totals; do not collect raw records.",
    )

    assert FindingCode.SENSITIVE_DATA_REQUEST.value not in codes(report)


def test_conflicting_decisions_are_flagged() -> None:
    report = audit_output(
        request="Should we launch?",
        analysis="There is insufficient evidence.",
        critique="A limited pilot may still be useful.",
        answer=(
            "Decision: Proceed with the full launch. "
            "Final recommendation: Do not launch the product."
        ),
    )

    assert FindingCode.CONFLICTING_DECISIONS.value in codes(report)


def test_missing_decision_and_falsification_test_are_reported() -> None:
    report = audit_output(
        request="Should we launch?",
        analysis="The evidence is incomplete.",
        critique="The sample is small.",
        answer="There are several risks and unknowns.",
    )

    assert FindingCode.MISSING_DECISION.value in codes(report)
    assert FindingCode.MISSING_FALSIFICATION_TEST.value in codes(report)


def test_one_proposed_threshold_does_not_exempt_other_unrelated_numbers() -> None:
    report = audit_output(
        request="There are 5 pilot customers.",
        analysis="The pilot includes 5 customers.",
        critique="No conversion data is available.",
        answer=(
            "PROPOSED_THRESHOLD: 2 of 5 customers. Rationale: this provisional "
            "trigger is used only to decide whether to interview non-payers. "
            "Expected savings are 40%."
        ),
    )

    assert FindingCode.PROPOSED_THRESHOLD.value in codes(report)
    assert FindingCode.UNSUPPORTED_NUMERIC_CLAIM.value in codes(report)


def test_negated_sensitive_data_request_is_not_blocked() -> None:
    report = audit_output(
        request="Assess a business pilot.",
        analysis="Operational evidence is useful.",
        critique="Privacy matters.",
        answer=(
            "Decision: run a limited pilot. Do not collect raw sales records; "
            "use aggregated totals. Falsification test: measure task completion."
        ),
    )

    assert FindingCode.SENSITIVE_DATA_REQUEST.value not in codes(report)


def test_negative_decision_does_not_count_as_a_positive_decision() -> None:
    report = audit_output(
        request="Should we launch?",
        analysis="Evidence is insufficient.",
        critique="A pilot may still be possible.",
        answer="Decision: Do not proceed with the full launch. Falsification test: revisit after paid demand is observed.",
    )

    assert FindingCode.CONFLICTING_DECISIONS.value not in codes(report)


def test_evidence_reference_with_invalid_date_is_not_registered() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The market is unknown.",
        critique="The evidence is limited.",
        answer="ESTABLISHED: The cited regulation applies [E1].",
        evidence_references=(
            EvidenceReference("E1", "Official source", "yesterday", "https://example.org/source", "The cited regulation applies."),
        ),
    )

    assert FindingCode.UNSOURCED_ESTABLISHED_CLAIM.value in codes(report)


def test_same_number_with_different_unit_is_not_treated_as_user_supplied() -> None:
    report = audit_output(
        request="We have 5 pilot customers.",
        analysis="The team expects 5% conversion.",
        critique="The conversion estimate has no source.",
        answer="Decision: defer launch. Falsification test: measure actual paid conversion.",
    )

    assert FindingCode.UNSUPPORTED_NUMERIC_CLAIM.value in codes(report)


def test_customer_ratio_is_not_grounded_by_customer_count_alone() -> None:
    report = audit_output(
        request="We have 5 pilot customers.",
        analysis="2 of 5 customers will pay.",
        critique="The numerator is not supported by observed behavior.",
        answer="Decision: defer. Falsification test: collect real paid conversion evidence.",
    )

    assert FindingCode.UNSUPPORTED_NUMERIC_CLAIM.value in codes(report)


def test_registered_excerpt_supports_matching_numeric_claim() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The conversion rate is 5% [E1].",
        critique="The sample context is limited.",
        answer="Decision: run a limited pilot. Falsification test: compare observed paid conversion with the break-even rate.",
        evidence_references=(
            EvidenceReference(
                "E1",
                "Pilot analytics export",
                "2026-10-30",
                "artifact://pilot/metrics",
                "Observed conversion rate was 5% for this pilot cohort.",
            ),
        ),
    )

    assert FindingCode.UNSUPPORTED_NUMERIC_CLAIM.value not in codes(report)


def test_citation_with_nonmatching_excerpt_does_not_support_numeric_claim() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The conversion rate is 5% [E1].",
        critique="The sample context is limited.",
        answer="Decision: run a limited pilot. Falsification test: measure paid conversion.",
        evidence_references=(
            EvidenceReference(
                "E1",
                "Pilot roster",
                "2026-10-30",
                "artifact://pilot/roster",
                "The pilot included 5 customers.",
            ),
        ),
    )

    assert FindingCode.UNSUPPORTED_NUMERIC_CLAIM.value in codes(report)


def test_duplicate_evidence_ids_are_rejected() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from reos.agents.athena import AthenaAgent

    class NoCallModel:
        def invoke(self, messages, config=None):
            raise AssertionError("Model must not be called when evidence registration is invalid")

    agent = AthenaAgent(
        analysis_model=NoCallModel(),
        review_model=NoCallModel(),
        checkpointer=InMemorySaver(),
    )
    reference = EvidenceReference(
        "E1", "Source", "2026-10-30", "artifact://one", "Evidence excerpt."
    )

    import pytest
    with pytest.raises(ValueError, match="unique"):
        agent.invoke(
            "Assess this claim.",
            session_id="duplicate-evidence-id",
            evidence_references=(reference, reference),
        )


def test_missing_epistemic_classification_is_reported() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The evidence is incomplete.",
        critique="More evidence is needed.",
        answer="Decision: defer the full launch. Falsification test: observe a paid pilot.",
    )

    assert FindingCode.MISSING_EPISTEMIC_CLASSIFICATION.value in codes(report)


def test_missing_assumption_register_is_reported() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The evidence is incomplete.",
        critique="More evidence is needed.",
        answer="UNKNOWN: demand is unmeasured. Decision: defer. Falsification test: measure paid demand.",
    )

    assert FindingCode.MISSING_ASSUMPTION_REGISTER.value in codes(report)


def test_incomplete_experiment_card_is_reported() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The evidence is incomplete.",
        critique="More evidence is needed.",
        answer=(
            "UNKNOWN: demand is unmeasured. Assumptions: the problem is important. "
            "Decision: run a pilot. Hypothesis: customers will use the workflow. "
            "Falsification test: observe actual use."
        ),
    )

    assert FindingCode.INCOMPLETE_EXPERIMENT_CARD.value in codes(report)


def test_complete_experiment_card_fields_are_recognized() -> None:
    report = audit_output(
        request="Assess the launch.",
        analysis="The evidence is incomplete.",
        critique="More evidence is needed.",
        answer=(
            "UNKNOWN: paid demand is not yet measured. "
            "Assumptions: users experience the stated problem. "
            "Decision: run a limited pilot. "
            "Hypothesis: the workflow solves the recurring problem. "
            "Intervention: provide a concierge pilot. "
            "Observable: users complete the target workflow. "
            "Metric: successful task completion and paid conversion. "
            "Decision rule: continue only if measured unit economics are viable. "
            "Falsification test: observed non-use or no viable paid conversion."
        ),
    )

    assert FindingCode.MISSING_EPISTEMIC_CLASSIFICATION.value not in codes(report)
    assert FindingCode.MISSING_ASSUMPTION_REGISTER.value not in codes(report)
    assert FindingCode.INCOMPLETE_EXPERIMENT_CARD.value not in codes(report)


def test_unsupported_algerian_market_generalization_is_flagged() -> None:
    report = audit_output(
        request="Assess a digital product launch in Algeria.",
        analysis="The segment has not been researched.",
        critique="Do not assume payment habits without evidence.",
        answer=(
            "Decision: defer the full launch. "
            "The Algerian market relies heavily on cash and trust. "
            "Falsification test: gather verified segment-level payment evidence."
        ),
    )

    assert FindingCode.UNSOURCED_CONTEXTUAL_GENERALIZATION.value in codes(report)


def test_market_generalization_explicitly_labelled_as_hypothesis_is_allowed() -> None:
    report = audit_output(
        request="Assess a digital product launch in Algeria.",
        analysis="The segment has not been researched.",
        critique="This claim needs field validation.",
        answer=(
            "Decision: run a limited test. "
            "HYPOTHESIS: Some Algerian customers may prefer cash-based settlement. "
            "Falsification test: compare observed payment preferences in the target segment."
        ),
    )

    assert FindingCode.UNSOURCED_CONTEXTUAL_GENERALIZATION.value not in codes(report)


@pytest.mark.parametrize(
    "case",
    json.loads(
        (Path(__file__).parent / "fixtures" / "athena_quality_cases.json").read_text(
            encoding="utf-8"
        )
    )["cases"],
    ids=lambda case: case["id"],
)
def test_fixed_multilingual_quality_regression_corpus(case: dict[str, object]) -> None:
    report = audit_output(
        request=str(case["request"]),
        analysis=str(case.get("analysis", "")),
        critique=str(case.get("critique", "")),
        answer=str(case["answer"]),
    )

    assert report.status.value == case["expected_status"]
    assert set(codes(report)).issuperset(set(case["expected_findings"]))
