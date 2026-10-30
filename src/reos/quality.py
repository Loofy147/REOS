from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
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
    MISSING_EPISTEMIC_CLASSIFICATION = "MISSING_EPISTEMIC_CLASSIFICATION"
    MISSING_ASSUMPTION_REGISTER = "MISSING_ASSUMPTION_REGISTER"
    INCOMPLETE_EXPERIMENT_CARD = "INCOMPLETE_EXPERIMENT_CARD"
    UNSOURCED_CONTEXTUAL_GENERALIZATION = "UNSOURCED_CONTEXTUAL_GENERALIZATION"


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """A source registered by the caller, not merely named by generated text."""

    reference_id: str
    source: str
    observed_at: str
    locator: str
    excerpt: str


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


_UNIT_PATTERN = (
    r"percent(?:age)?|%|٪|days?|weeks?|months?|years?|hours?|minutes?|"
    r"users?|customers?|clients?|interviews?|sales?|records?|units?|"
    r"\$|€|USD|EUR|DZD|dollars?|euros?|dinars?|دج|دينار|يوم|أيام|يوماً|يومًا|أسبوع|أسابيع|أسبوعين|"
    r"شهر|أشهر|ساعة|ساعات|عميل|عملاء|مستخدم|مستخدمين|مقابلة|مقابلات|سجل|سجلات|مبيعات"
)
_RATIO_RE = re.compile(
    rf"(?P<numerator>\d+)\s*(?:out\s+of|of|من\s+أصل|من)\s*"
    rf"(?P<denominator>\d+)\s*(?P<unit>{_UNIT_PATTERN})(?![\w])",
    re.IGNORECASE,
)
_NUMBER_UNIT_RE = re.compile(
    rf"(?P<value>\d+(?:[.,]\d+)?)\s*[-–]?\s*"
    rf"(?:[A-Za-z]+\s+){{0,2}}(?P<unit>{_UNIT_PATTERN})(?![\w])",
    re.IGNORECASE,
)
_CURRENCY_SCALE_RE = re.compile(
    r"(?:(?P<prefix>\$|€|USD|EUR|DZD|دج|دينار)\s*)?"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<scale>thousand|million|billion|ألف|آلاف|مليون|ملايين|مليار|مليارات)\s*"
    r"(?P<suffix>USD|EUR|DZD|dollars?|euros?|dinars?|دج|دينار|\$|€)?",
    re.IGNORECASE,
)
_PREFIX_CURRENCY_RE = re.compile(
    r"(?P<unit>\$|€|USD|EUR|DZD|دج|دينار)\s*"
    r"(?P<value>\d+(?:[.,]\d+)?)(?![\w])",
    re.IGNORECASE,
)
_METRIC_LABEL_PATTERN = (
    r"false\\s+positive\\s+rate|false\\s+negative\\s+rate|"
    r"conversion(?:\\s+rate)?|precision|recall|accuracy|f1(?:\\s+score)?|"
    r"fpr|fnr|cac|roi|margin|retention|churn|probability|risk\\s+score|"
    r"معدل\\s+الإيجابيات\\s+الكاذبة|معدل\\s+السلبيات\\s+الكاذبة|"
    r"معدل\\s+التحويل|نسبة\\s+التحويل|الدقة|الاستدعاء|الهامش|الاحتفاظ|التسرب|الاحتمال|درجة\\s+المخاطر"
)
_BARE_METRIC_RE = re.compile(
    rf"(?:(?P<label>{_METRIC_LABEL_PATTERN})\\s*(?:mean|average|is|was|of|=|:|بلغ|يساوي|وصل\\s+إلى)?\\s*"
    rf"(?P<value>\\d+(?:[.,]\\d+)?)|"
    rf"(?P<value_after>\\d+(?:[.,]\\d+)?)\\s*(?:for|in|of)?\\s*"
    rf"(?P<label_after>{_METRIC_LABEL_PATTERN}))",
    re.IGNORECASE,
)
_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
_SCALE_FACTORS = {
    "thousand": Decimal("1000"),
    "ألف": Decimal("1000"),
    "آلاف": Decimal("1000"),
    "million": Decimal("1000000"),
    "مليون": Decimal("1000000"),
    "ملايين": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "مليار": Decimal("1000000000"),
    "مليارات": Decimal("1000000000"),
}

_SENTENCE_RE = re.compile(r"(?<=[.!?؟؛])\s+|\n+")
_REFERENCE_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]{0,31})\]")
_ESTABLISHED_RE = re.compile(r"\bESTABLISHED\s*[:：]", re.IGNORECASE)
_USER_REPORTED_RE = re.compile(r"\b(?:USER_REPORTED|HYPOTHESIS|UNKNOWN)\s*[:：]", re.IGNORECASE)
_CONTEXT_MARKET_RE = re.compile(
    r"(?:في\s+الجزائر|السوق\s+الجزائري|السوق\s+المحلي|الاقتصاد\s+الجزائري|"
    r"المستهلك\s+الجزائري|العملاء\s+الجزائريون|الجزائريون|Algeria|"
    r"Algerian\s+(?:market|consumers|customers))",
    re.IGNORECASE,
)
_GENERALIZATION_RE = re.compile(
    r"(?:يعتمد(?:\s+بشكل\s+كبير)?\s+على|يغلب\s+عليه|غالباً|غالبًا|عادةً|عادة|"
    r"معظم|الأغلبية|منتشر|شائع|cash[- ]based|mostly|predominantly|typically|"
    r"generally|relies\s+heavily\s+on|most\s+customers|widespread)",
    re.IGNORECASE,
)
_EPISTEMIC_LABEL_RE = re.compile(
    r"\b(?:ESTABLISHED|USER_REPORTED|INFERENCE|HYPOTHESIS|UNKNOWN)\b",
    re.IGNORECASE,
)
_ASSUMPTION_RE = re.compile(
    r"(?:assumptions?|الافتراضات|الافتراض|فرضيات|الفرضيات)",
    re.IGNORECASE,
)
_EXPERIMENT_FIELDS = {
    "hypothesis": re.compile(r"(?:\bhypothesis\b|\bفرضية\b|\bالفرضية\b)", re.I),
    "intervention": re.compile(r"(?:\bintervention\b|\baction\b|\bالتدخل\b|\bالإجراء\b)", re.I),
    "observation": re.compile(r"(?:\bobservable\b|\bobservation\b|\boutcome\b|\bالملاحظة\b|\bالنتيجة\b)", re.I),
    "metric": re.compile(r"(?:\bmetric\b|\bmeasure\b|\bmeasurement\b|\bمقياس\b|\bالقياس\b)", re.I),
    "decision_rule": re.compile(r"(?:\bdecision\s+rule\b|\bdecision\s+criterion\b|\bقاعدة\s+القرار\b|\bمعيار\s+القرار\b)", re.I),
}
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
    r"(?:\bproceed\b|\bgo ahead\b|\bapprove\b|\bbuild\b|"
    r"\bstart\b|\bbegin\b|\brelease\b|\bgo live\b|"
    r"المضي\s+قدماً|المضي\s+قدمًا|اعتماد|ابدأ|استمر)",
    re.IGNORECASE,
)
_NEGATED_ACTION_RE = re.compile(
    r"(?:\b(?:do not|don't|never)\s+(?:proceed|go ahead|launch|approve|build)\b|"
    r"لا\s+(?:تطلق|تبدأ|تعتمد|تمضِ|توافق))",
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
    r"identifiable\s+(?:customer|supplier|financial)\s+(?:records?|data)|"
    r"supplier\s+lists?|customer\s+lists?|bank\s+(?:statements?|details?)|account\s+(?:numbers?|details?)|"
    r"identity\s+documents?|سجلات\s+مبيعات\s+(?:خام|كاملة|حقيقية)|"
    r"قائمة\s+الموردين|قوائم\s+الموردين|بيانات\s+العملاء\s+الكاملة|"
    r"كشوف\s+الحساب|أرقام\s+الحسابات|وثائق\s+الهوية)",
    re.IGNORECASE,
)
_REQUEST_VERB_RE = re.compile(
    r"(?:require|ask|request|upload|send|share|provide|collect|obtain|gather|store|retain|capture|record|demand|"
    r"اطلب|اطلبوا|اشترط|ارفع|أرسل|ارسل|شارك|زوّد|زود|اجمع|احتفظ|خزّن|خزن|سجّل|سجل)",
    re.IGNORECASE,
)
_NEGATED_REQUEST_RE = re.compile(
    r"(?:\b(?:do not|don't|never|avoid|not to)\s+(?:require|ask|request|upload|send|"
    r"share|provide|collect|obtain|gather|store|retain|capture|record|demand)\b|"
    r"لا\s+(?:تطلب|تجمع|ترفع|ترسل|تشارك|تشترط|تزود|تحتفظ|تخزن|تسجل))",
    re.IGNORECASE,
)


def _sentences(text: str) -> Iterable[str]:
    """Split prose without deleting substantive leading numbers such as '2 of 5'."""
    for item in _SENTENCE_RE.split(text):
        sentence = item.strip(" \t-*•")
        sentence = re.sub(r"^\(?\d{1,3}[.)]\s+", "", sentence)
        sentence = sentence.lstrip(" :")
        if sentence:
            yield sentence


def _canonical_unit(unit: str) -> str:
    lowered = unit.lower()
    if lowered in {"%", "٪", "percent", "percentage"}:
        return "percent"
    if lowered in {"day", "days", "يوم", "أيام", "يوماً", "يومًا"}:
        return "days"
    if lowered in {"week", "weeks", "أسبوع", "أسابيع", "أسبوعين"}:
        return "weeks"
    if lowered in {"month", "months", "شهر", "أشهر"}:
        return "months"
    if lowered in {"year", "years"}:
        return "years"
    if lowered in {"hour", "hours", "ساعة", "ساعات"}:
        return "hours"
    if lowered in {"minute", "minutes"}:
        return "minutes"
    if lowered in {"customer", "customers", "client", "clients", "عميل", "عملاء"}:
        return "customers"
    if lowered in {"user", "users", "مستخدم", "مستخدمين"}:
        return "users"
    if lowered in {"interview", "interviews", "مقابلة", "مقابلات"}:
        return "interviews"
    if lowered in {"record", "records", "سجل", "سجلات"}:
        return "records"
    if lowered in {"دج", "دينار", "dinar", "dinars", "dzd"}:
        return "DZD"
    if lowered in {"$", "usd", "dollar", "dollars"}:
        return "USD"
    if lowered in {"€", "eur", "euro", "euros"}:
        return "EUR"
    return lowered


def _decimal_string(value: Decimal) -> str:
    normalized = value.normalize()
    return format(normalized, "f")


def _scaled_currency(value: str, scale: str) -> str:
    factor = _SCALE_FACTORS.get(scale.lower(), Decimal("1"))
    return _decimal_string(Decimal(value.replace(",", ".")) * factor)


def _metric_tokens(text: str) -> set[str]:
    """Extract numeric quantities with units, currency scales, ratios, and metric labels."""
    text = text.translate(_DIGIT_TRANSLATION)
    claims: set[str] = set()
    for match in _BARE_METRIC_RE.finditer(text):
        label = match.group("label") or match.group("label_after")
        value = match.group("value") or match.group("value_after")
        normalized_label = re.sub(r"\\s+", "_", label.lower())
        normalized_value = value.replace(",", ".")
        claims.add(f"metric:{normalized_label}={normalized_value}")
    ratio_spans: list[tuple[int, int]] = []
    currency_scale_spans: list[tuple[int, int]] = []
    prefix_currency_spans: list[tuple[int, int]] = []

    for match in _RATIO_RE.finditer(text):
        ratio_spans.append(match.span())
        unit = _canonical_unit(match.group("unit"))
        claims.add(
            f"ratio:{match.group('numerator')}/{match.group('denominator')}:{unit}"
        )

    for match in _CURRENCY_SCALE_RE.finditer(text):
        currency = match.group("prefix") or match.group("suffix")
        if not currency:
            continue
        currency_scale_spans.append(match.span())
        unit = _canonical_unit(currency)
        value = _scaled_currency(match.group("value"), match.group("scale"))
        claims.add(f"{value}:{unit}")

    for match in _PREFIX_CURRENCY_RE.finditer(text):
        if any(
            match.start() < end and match.end() > start
            for start, end in currency_scale_spans
        ):
            continue
        prefix_currency_spans.append(match.span())
        unit = _canonical_unit(match.group("unit"))
        value = match.group("value").replace(",", ".")
        claims.add(f"{value}:{unit}")

    for match in _NUMBER_UNIT_RE.finditer(text):
        if any(
            match.start() < end and match.end() > start
            for start, end in ratio_spans + currency_scale_spans + prefix_currency_spans
        ):
            continue
        value = match.group("value").replace(",", ".")
        unit = _canonical_unit(match.group("unit"))
        claims.add(f"{value}:{unit}")
    return claims


def _matching_references(
    sentence: str, references: tuple[EvidenceReference, ...]
) -> tuple[EvidenceReference, ...]:
    """Return only cited references with a date, locator, and retrieved excerpt."""
    cited_ids = {match.group(1) for match in _REFERENCE_RE.finditer(sentence)}
    matched: list[EvidenceReference] = []
    for reference in references:
        if reference.reference_id not in cited_ids:
            continue
        if not (
            reference.source.strip()
            and reference.locator.strip()
            and reference.excerpt.strip()
        ):
            continue
        try:
            date.fromisoformat(reference.observed_at)
        except ValueError:
            continue
        matched.append(reference)
    return tuple(matched)


def _verified_references(
    sentence: str, references: tuple[EvidenceReference, ...]
) -> bool:
    """Check registry linkage, not source authenticity or semantic entailment."""
    return bool(_matching_references(sentence, references))


def _numeric_claim_supported_by_reference(
    sentence: str, references: tuple[EvidenceReference, ...]
) -> bool:
    claims = _metric_tokens(sentence)
    if not claims:
        return True
    return any(
        claims.issubset(_metric_tokens(reference.excerpt))
        for reference in _matching_references(sentence, references)
    )


def _is_proposed_threshold(sentences: list[str], index: int) -> bool:
    current = sentences[index]
    previous = sentences[index - 1] if index else ""
    following = sentences[index + 1] if index + 1 < len(sentences) else ""
    has_label = bool(_PROPOSED_RE.search(current) or _PROPOSED_RE.search(previous))
    has_rationale = bool(
        _RATIONALE_RE.search(current) or _RATIONALE_RE.search(following)
    )
    return has_label and has_rationale


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
    supplied_metrics = _metric_tokens(request)
    stages = (("analysis", analysis), ("review", critique), ("synthesis", answer))

    for stage_name, text in stages:
        sentences = list(_sentences(text))
        for index, sentence in enumerate(sentences):
            metrics = _metric_tokens(sentence)
            if not metrics:
                continue
            if metrics.issubset(supplied_metrics):
                continue
            if _numeric_claim_supported_by_reference(sentence, evidence_references):
                continue
            if _is_proposed_threshold(sentences, index):
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
        if (
            _CONTEXT_MARKET_RE.search(sentence)
            and _GENERALIZATION_RE.search(sentence)
            and not _USER_REPORTED_RE.search(sentence)
            and not _verified_references(sentence, evidence_references)
        ):
            findings.append(
                QualityFinding(
                    FindingCode.UNSOURCED_CONTEXTUAL_GENERALIZATION,
                    QualityStatus.WARN,
                    "A contextual or market-wide generalization is stated without a registered source. Downgrade it to HYPOTHESIS or UNKNOWN, or provide the inspected supporting excerpt.",
                    sentence[:240],
                )
            )

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
            and not _NEGATED_REQUEST_RE.search(sentence)
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
        _POSITIVE_DECISION_RE.search(_NEGATED_ACTION_RE.sub("", sentence))
        for sentence in decision_sentences
    )
    has_negative = any(
        _NEGATIVE_DECISION_RE.search(sentence)
        for sentence in decision_sentences
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

    if not _EPISTEMIC_LABEL_RE.search(answer):
        findings.append(
            QualityFinding(
                FindingCode.MISSING_EPISTEMIC_CLASSIFICATION,
                QualityStatus.WARN,
                "The final answer does not explicitly classify material claims as ESTABLISHED, USER_REPORTED, INFERENCE, HYPOTHESIS, or UNKNOWN.",
            )
        )

    if not _ASSUMPTION_RE.search(answer):
        findings.append(
            QualityFinding(
                FindingCode.MISSING_ASSUMPTION_REGISTER,
                QualityStatus.WARN,
                "The final answer does not include an explicit assumption register. State the assumptions that materially affect the decision.",
            )
        )

    missing_experiment_fields = [
        field
        for field, pattern in _EXPERIMENT_FIELDS.items()
        if not pattern.search(answer)
    ]
    if missing_experiment_fields:
        findings.append(
            QualityFinding(
                FindingCode.INCOMPLETE_EXPERIMENT_CARD,
                QualityStatus.WARN,
                "The falsification test is missing experiment-card fields: "
                + ", ".join(missing_experiment_fields)
                + ". Include hypothesis, intervention, observable, metric, and decision rule; mark unavailable fields UNKNOWN rather than inventing them.",
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
