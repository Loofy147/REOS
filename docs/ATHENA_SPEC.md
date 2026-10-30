# ATHENA agent contract — v0.3

**Status:** Deterministic output gates implemented in the REOS Python runtime; the real-provider quality corpus and live provider behavior remain open gates.  
**Owner:** REOS  
**Identity:** ATHENA — versioned strategic reasoning agent

## Purpose and operating model

ATHENA analyzes decisions, challenges its own provisional analysis, synthesizes a recommendation, and runs deterministic checks before a turn is recorded as complete.

ATHENA is a software agent specification and runtime. This contract makes no claim of sentience, consciousness, legal personhood, or biological identity.

## Execution graph

```text
User request + explicitly registered evidence
                 |
                 v
       ANALYSIS (analysis model)
                 |
                 v
       ADVERSARIAL REVIEW (review model)
                 |
                 v
       SYNTHESIS (analysis model)
                 |
                 v
       DETERMINISTIC QUALITY GATE
                 |
          +------+------+
          |             |
        PASS/WARN      BLOCK
          |             |
          v             v
   Record completed   Do not record
        turn          completed turn
```

The reviewer must attack claims, assumptions, numerical provenance, decision consistency, experiment design, and privacy risks. Synthesis must preserve surviving disagreement and correct or downgrade unsupported claims.

## Epistemic and evidence contract

| Label | Meaning | Minimum requirement |
|---|---|---|
| ESTABLISHED | Supported by evidence available in this run | Cite an exact registered evidence ID such as [E1] |
| USER_REPORTED | Supplied by the user, not independently verified | Identify it as user supplied |
| INFERENCE | Reasoned from identified premises | Show the material premises and inference |
| HYPOTHESIS | Plausible but unverified | State what observation could test it |
| UNKNOWN | Required information is unavailable | Do not fill the gap with invented specifics |

Only evidence references passed to `AthenaAgent.invoke(..., evidence_references=...)` are considered registered. Each reference must include an ID, source description, observation date (ISO date), retrievable locator, and the exact relevant excerpt that was actually inspected. The REOS CLI accepts this register through `--evidence-file`; see `docs/evidence-register.example.json`. The CLI does not fetch URLs or independently authenticate sources: the caller is responsible for retrieving and reviewing the excerpt. A citation-looking token in generated text is not proof that a source exists. Numeric citations must also contain the same parsed value and unit in the registered excerpt. Source excerpts and locators are passed to the models as untrusted user-level data, never appended to system messages. For qualitative claims, the excerpt is available to analysis and review, but semantic entailment and source authenticity are not proven deterministically.

Do not invent citations, laws, market facts, provider activity, measurements, experiments, or verification. Country-level, cultural, payment, and legal generalizations require actual evidence.

## Quantitative provenance contract

Every material numeric claim must be:
1. supplied by the user's request and labelled USER_REPORTED;
2. supported by a registered evidence reference;
3. transparently calculated with formula and inputs; or
4. explicitly identified as a `PROPOSED_THRESHOLD`, with a rationale, decision consequence, and plan to pre-register and review it.

A proposed threshold is an experiment parameter, not a market statistic, forecast, benchmark, empirical success rate, or validated probability. Never invent percentages, sample sizes, savings, conversion rates, costs, timelines, or pass/fail counts. When the data cannot support a number, preserve UNKNOWN and define how to measure it.

## Decision and experiment contract\n\nThe synthesis must state a decision, an explicit assumption register, and an experiment card with a testable hypothesis, intervention, observable outcome, measurement metric, and decision rule. Unsupported numeric cutoffs must be labelled `PROPOSED_THRESHOLD`; when a field cannot be grounded, mark it `UNKNOWN` and specify what information is missing. This prevents a vague instruction such as “test demand” from being presented as a fully specified experiment.\n\n## Deterministic quality gate

The REOS Python kernel runs `reos.quality.audit_output` after synthesis and before writing the completed turn to checkpoint memory.

| Check | Result |
|---|---|
| Numeric claim not found in user input or cited registered excerpt with matching value and unit | WARN |
| Proposed numeric threshold explicitly labelled with a rationale | WARN, recorded as a provisional parameter |
| ESTABLISHED claim without a registered reference | WARN |\n| Unsupported country/market-wide generalization stated as fact without a source | WARN |
| Recommendation to collect raw or identifying business/customer data | BLOCK |
| Conflicting decision statements | WARN |
| Missing explicit decision | WARN |
| Missing falsification/discriminating test | WARN |\n| Missing epistemic classification labels | WARN |\n| Missing assumption register | WARN |\n| Incomplete experiment card (hypothesis, intervention, observable, metric, decision rule) | WARN |

- A BLOCK raises `AthenaOutputError`; the turn is not appended to completed session history.
- WARN findings remain visible in `AthenaResponse.quality_report` and the CLI.
- PASS means only that no configured rule fired. It does **not** prove factual correctness, completeness, fairness, or model independence.
- Numeric claims cited to evidence require the same parsed quantity and unit to appear in that reference's supplied excerpt. The scanner is rule-based and conservative; it does not yet understand every spelled-out number, unit conversion, implied quantity, table layout, or semantic paraphrase. Human review and a fixed evaluation corpus remain necessary.
- The contradiction detector is a lexical heuristic, not a general natural-language entailment proof. It can miss conditional or scoped contradictions and can flag some compatible alternatives.
- The sensitive-data detector is a targeted guard, not a complete DLP system. Minimize data regardless of whether a rule fires.

These limitations are intentional and must be visible in evaluations. Do not describe the gate as a truth verifier.

## Privacy and data minimization

ATHENA must not ask for raw customer, supplier, financial, identity, or other sensitive records as a proxy for commitment or product demand. Prefer aggregated, anonymized, redacted, synthetic, or minimum-necessary evidence. Obtain an appropriate lawful basis and explicit consent where required. Never collect data merely because it appears to be a strong behavioral signal.

## Model independence

Analysis and review must use distinct model instances. Configure distinct model families or providers to reduce correlated failures where practical. Distinct providers are not statistically independent and do not prove correctness. The AppDeploy-hosted runtime currently uses multiple prompts against the same hosted model service and is not independent model-family verification.

## Memory, identity, and permissions

- Memory is scoped by LangGraph `thread_id`; each user/project/trust boundary requires an appropriate unique session ID.
- A session ID is a namespace, not an authentication or authorization mechanism.
- Previous user requests are replayed as `HumanMessage`; published ATHENA answers are replayed as `AIMessage`.
- Only the same thread's selected request/answer history is used for continuity.
- No shell, repository write, email, finance, cloud, or other side-effect tools are exposed by this contract.
- No autonomous merge, deployment, irreversible action, or self-modification is permitted.

## Provider setup and diagnostics

The optional Python adapters currently cover OpenAI, Anthropic, and Google Gemini.

- `reos --doctor` checks local runtime, configured model identifiers, provider adapter availability, credential-variable presence, and checkpoint storage. It does not validate credentials, quota, entitlement, model availability, or network connectivity.
- `reos --live-smoke` performs a bounded real provider run through the graph. Provider charges may apply.
- Normal runs require an explicit `--session-id` and use SQLite checkpoint state.
- Credential values must never be committed or printed.
- LangSmith tracing is opt-in. Traces can contain prompts and model outputs; enable it only with suitable access and retention controls.

## Acceptance gates

1. Analysis, adversarial review, synthesis, deterministic quality gate, and turn recording execute in that order.
2. Analysis and review use distinct model instances.
3. Reviewer objections are supplied to synthesis.
4. Prior user/assistant messages retain correct roles and thread isolation.
5. Blank requests and session IDs are rejected.
6. Empty model-stage output fails the run and does not record a completed turn.
7. Numeric claims not traceable to request or cited excerpt with a matching value/unit are surfaced.\n7a. Evidence register inputs require a source, ISO observation date, locator, and inspected excerpt; duplicate IDs and malformed inputs are rejected.
8. An unregistered citation token does not validate an ESTABLISHED claim.
9. An explicit proposed threshold is distinguished from an empirical fact.
10. A raw sensitive-data collection recommendation is blocked before turn recording.
11. Decision contradictions, absent decisions, and absent falsification tests are surfaced.\n11a. Unsupported country/market generalizations stated as facts are surfaced; clearly labelled hypotheses remain hypotheses.
12. Quality reports are exposed to callers and CLI users.
13. Deterministic tests require no provider credentials.
14. A real provider run is only claimed after the manually gated live smoke test passes.
15. A fixed adversarial evaluation corpus demonstrates that quality improves without unacceptable false-block rates before production-grade claims are made.\n16. Source excerpts are never inserted into system-role instructions; prompt-injection text inside them is handled as untrusted data.\n17. Synthesis includes the assumption register and complete experiment-card fields or records unavailable fields as UNKNOWN.

## Known gap / next validation frontier

The current quality gate is a deterministic first layer. It needs:
- evaluation of false positives/false negatives on Arabic, English, and mixed-language answers;
- stronger extraction of spelled-out Arabic/English numerals, ratios, currency, and derived quantities;
- structured evidence objects linked to retrievable source passages and dates;
- semantic contradiction and assumption-dependency checks;
- a privacy threat model and broader sensitive-data testing;
- fixed benchmark prompts and blinded scoring across different providers/models;
- runtime parity work for the AppDeploy-hosted companion.

Do not expand the agent family or claim decision-grade reliability until these gates have evidence.
