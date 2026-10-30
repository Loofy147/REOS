from __future__ import annotations

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
            EvidenceReference("E1", "Official source", "2026-10-30", "https://example.org/source"),
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
            EvidenceReference("E1", "Official source", "2026-10-30", ""),
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
            EvidenceReference("E1", "Official source", "yesterday", "https://example.org/source"),
        ),
    )

    assert FindingCode.UNSOURCED_ESTABLISHED_CLAIM.value in codes(report)
