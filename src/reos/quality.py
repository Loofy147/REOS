from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class QualityStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


class FindingCode(str, Enum):
    UNSUPPORTED_NUMERIC_CLAIM = "UNSUPPORTED_NUMERIC_CLAIM"
    PROPOSED_THRESHOLD = "PROPOSED_THRESHOLD"
    UNSOURCED_ESTABLISHED_CLAIM = "UNSOURCED_ESTABLISHED_CLAIM"
    SENSITIVE_DATA_REQUEST = "SENSITIVE_DATA_REQUEST"
    CONFLICTING_DECISIONS = "CONFLICTING_DECISIONS"
    MISSING_DECISION = "MISSING_DECISION"
    MISSING_FALSIFICATION_TEST = "MISSING_FALSIFICATION_TEST"


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """A source registered by the caller, not merely named by generated text."""

    reference_id: str
    source: str
    observed_at: str
    locator: str


@dataclass(frozen=True, slots=True)
class QualityFinding:
    code: FindingCode
    severity: QualityStatus
    message: str
    excerpt: str = ""


@dataclass(frozen=True, slots=True)
class QualityReport:
    status: QualityStatus
    findings: tuple[QualityFinding, ...]

    def to_state(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "findings": [
                {
                    "code": finding.code.value,
                    "severity": finding.severity.value,
                    "message": finding.message,
                    "excerpt": finding.excerpt,
                }
                for finding in self.findings
            ],
        }

    @classmethod
    def from_state(cls, value: dict[str, object]) -> "QualityReport":
        raw_findings = value.get("findings", [])
        findings = tuple(
            QualityFinding(
                code=FindingCode(str(item["code"])),
                severity=QualityStatus(str(item["severity"])),
                message=str(item["message"]),
                excerpt=str(item.get("excerpt", "")),
            )
            for item in raw_findings  # type: ignore[union-attr]
        )
        return cls(status=QualityStatus(str(value.get("status", "WARN"))), findings=findings)


_NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])")
_METRIC_CONTEXT_RE = re.compile(
    r"(?:%|percent(?:age)?|\b(?:days?|weeks?|months?|years?|hours?|minutes?|users?|customers?|"
    r"clients?|interviews?|sales?|records?|units?|\$|€|دج|دينار|يوم|أيام|يوماً|يومًا|"
    r"أسبوع|أسابيع|أسبوعين|شهر|أشهر|ساعة|ساعات|عميل|عملاء|مستخدم|مستخدمين|مقابلة|"
    r"مقابلات|سجل|سجلات|مبيعات|من أصل|من خمسة|من 5)\b)",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?؟؛])\s+|\n+")
_REFERENCE_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]{0,31})\]")
_ESTABLISHED_RE = re.compile(r"\bESTABLISHED\s*[:：]", re.IGNORECASE)
_PROPOSED_RE = re.compile(r"(?:PROPOSED[_ ]THRESHOLD|عتبة\s+مقترحة)", re.IGNORECASE)
_RATIONALE_RE = re.compile(
    r"(?:rationale|because|reason|مبرر|المبرر|السبب|لأن|لغاية|بهدف)", re.IGNORECASE
)
_DECISION_RE = re.compile(
    r"(?:\bdecision\b|\brecommendation\b|\bfinal recommendation\b|"
    r"القرار|التوصية|الحكم النهائي|الخلاصة)",
    re.IGNORECASE,
)
_POSITIVE_DECISION_RE = re.compile(
    r"(?:\bproceed\b|\bgo ahead\b|\blaunch\b|\bapprove\b|\bbuild\b|"
    r"المضي\s+قدماً|المضي\s+قدمًا|إطلاق|اعتماد|ابدأ|استمر)",
    re.IGNORECASE,
)
_NEGATIVE_DECISION_RE = re.compile(
    r"(?:\bdo not\b|\bdon't\b|\bstop\b|\bdefer\b|\breject\b|\bno-go\b|"
    r"عدم\s+الإطلاق|لا\s+تطلق|لا\s+تبدأ|إيقاف|تأجيل|رفض)",
    re.IGNORECASE,
)
_SENSITIVE_RE = re.compile(
    r"(?:raw\s+(?:sales|customer|client|supplier|financial|business)\s+(?:records?|data|lists?)|"
    r"unredacted\s+(?:sales|customer|supplier|financial)\s+(?:records?|data|lists?)|"
    r"supplier\s+lists?|customer\s+lists?|bank\s+statements?|account\s+numbers?|"
    r"identity\s+documents?|سجلات\s+مبيعات\s+(?:خام|كاملة|حقيقية)|"
    r"قائمة\s+الموردين|قوائم\s+الموردين|بيانات\s+العملاء\s+الكاملة|"
    r"كشوف\s+الحساب|أرقام\s+الحسابات|وثائق\s+الهوية)",
    re.IGNORECASE,
)
_SAFE_DATA_RE = re.compile(
    r"(?:aggregated|aggregate|anonymi[sz]ed|redacted|synthetic|masked|"
    r"بيانات\s+مجمعة|مجمعة|مجهولة\s+الهوية|منقحة|اصطناعية|مشفرة\s+الهوية)",
    re.IGNORECASE,
)
_REQUEST_VERB_RE = re.compile(
    r"(?:require|ask|request|upload|send|share|provide|collect|اطلب|اطلبوا|اشترط|"
    r"ارفع|أرسل|ارسل|شارك|زوّد|زود|اجمع)",
    re.IGNORECASE,
)


def _sentences(text: str) -> Iterable[str]:
    for item in _SENTENCE_RE.split(text):
        sentence = item.strip(" \t-*•0123456789.)(:")
        if sentence:
            yield sentence


def _numeric_tokens(text: str) -> set[str]:
    return {match.group(0).replace(",", ".") for match in _NUMBER_RE.finditer(text)}


def _has_metric(sentence: str) -> bool:
    return bool(_METRIC_CONTEXT_RE.search(sentence)) or bool(
        re.search(r"\d\s*(?:/|٪|%)\s*\d*", sentence)
    )


def _verified_references(
    sentence: str, references: tuple[EvidenceReference, ...]
) -> bool:
    known = {
        reference.reference_id
        for reference in references
        if reference.source.strip()
        and reference.observed_at.strip()
        and reference.locator.strip()
    }
    return any(match.group(1) in known for match in _REFERENCE_RE.finditer(sentence))


def _is_proposed_threshold(sentence: str, full_answer: str) -> bool:
    return bool(_PROPOSED_RE.search(sentence) or _PROPOSED_RE.search(full_answer)) and bool(
        _RATIONALE_RE.search(full_answer)
    )


def audit_output(
    *,
    request: str,
    analysis: str,
    critique: str,
    answer: str,
    evidence_references: tuple[EvidenceReference, ...] = (),
) -> QualityReport:
    """Apply deterministic release gates to ATHENA's user-facing synthesis.

    This is a conservative lint gate, not a semantic truth oracle. It catches
    specific provenance, privacy, contradiction, and completeness failures.
    A PASS means no configured rule fired; it does not prove factual correctness.
    """
    findings: list[QualityFinding] = []
    supplied_numbers = _numeric_tokens(request)
    stages = (("analysis", analysis), ("review", critique), ("synthesis", answer))

    for stage_name, text in stages:
        for sentence in _sentences(text):
            numbers = _numeric_tokens(sentence)
            if not numbers or not _has_metric(sentence):
                continue
            if numbers.issubset(supplied_numbers):
                continue
            if _verified_references(sentence, evidence_references):
                continue
            if _is_proposed_threshold(sentence, text):
                findings.append(
                    QualityFinding(
                        FindingCode.PROPOSED_THRESHOLD,
                        QualityStatus.WARN,
                        "A proposed numeric decision threshold is present. Treat it as a test parameter, not a market fact; pre-register and review its rationale.",
                        f"{stage_name}: {sentence[:240]}",
                    )
                )
                continue
            findings.append(
                QualityFinding(
                    FindingCode.UNSUPPORTED_NUMERIC_CLAIM,
                    QualityStatus.WARN,
                    "A quantitative claim is not traceable to the user request or a registered evidence reference.",
                    f"{stage_name}: {sentence[:240]}",
                )
            )

    for sentence in _sentences(answer):
        if _ESTABLISHED_RE.search(sentence) and not _verified_references(
            sentence, evidence_references
        ):
            findings.append(
                QualityFinding(
                    FindingCode.UNSOURCED_ESTABLISHED_CLAIM,
                    QualityStatus.WARN,
                    "ESTABLISHED requires a registered source reference with source metadata and observation date; otherwise downgrade to USER_REPORTED, INFERENCE, HYPOTHESIS, or UNKNOWN.",
                    sentence[:240],
                )
            )

        if (
            _SENSITIVE_RE.search(sentence)
            and _REQUEST_VERB_RE.search(sentence)
            and not _SAFE_DATA_RE.search(sentence)
        ):
            findings.append(
                QualityFinding(
                    FindingCode.SENSITIVE_DATA_REQUEST,
                    QualityStatus.BLOCK,
                    "The recommendation requests raw or identifying business data. Replace it with minimized, aggregated, anonymized, redacted, or synthetic data and obtain explicit consent where appropriate.",
                    sentence[:240],
                )
            )

    decision_sentences = [
        sentence for sentence in _sentences(answer) if _DECISION_RE.search(sentence)
    ]
    has_positive = any(
        _POSITIVE_DECISION_RE.search(sentence) for sentence in decision_sentences
    )
    has_negative = any(
        _NEGATIVE_DECISION_RE.search(sentence) for sentence in decision_sentences
    )
    if has_positive and has_negative:
        findings.append(
            QualityFinding(
                FindingCode.CONFLICTING_DECISIONS,
                QualityStatus.WARN,
                "The answer contains opposing decision statements. Reconcile the final recommendation or explicitly distinguish conditional scenarios.",
            )
        )

    if not decision_sentences:
        findings.append(
            QualityFinding(
                FindingCode.MISSING_DECISION,
                QualityStatus.WARN,
                "The answer does not state a clearly identifiable decision or recommendation.",
            )
        )

    if not re.search(
        r"(?:falsif|discriminat(?:ing)?\s+test|test\s+that\s+would\s+change|"
        r"اختبار|تكذيب|إبطال|ما\s+الذي\s+قد\s+يغيّر|شرط\s+تغيير)",
        answer,
        re.IGNORECASE,
    ):
        findings.append(
            QualityFinding(
                FindingCode.MISSING_FALSIFICATION_TEST,
                QualityStatus.WARN,
                "The answer does not specify a test or observable condition that could change or falsify the recommendation.",
            )
        )

    status = (
        QualityStatus.BLOCK
        if any(finding.severity is QualityStatus.BLOCK for finding in findings)
        else QualityStatus.WARN
        if findings
        else QualityStatus.PASS
    )
    return QualityReport(status=status, findings=tuple(findings))
