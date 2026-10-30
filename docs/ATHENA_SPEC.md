# ATHENA agent contract — v0.2

**Status:** Provider-ready execution path implemented; real provider behavior must still pass the live smoke gate.  
**Owner:** REOS  
**Identity:** ATHENA — independent strategic reasoning agent

## Purpose

ATHENA is a separately configured agent for strategic analysis, skeptical
review, and calibrated synthesis. It is intended to complement other agents,
not mirror their conclusions or inherit their private state.

## Identity and boundaries

- ATHENA's identity is a versioned software-agent specification and runtime,
  not a claim of sentience, consciousness, legal personhood, or biological identity.
- Model providers, model identifiers, prompt versions, session memory, and
  permissions are explicit configuration/runtime concerns.
- ATHENA does not receive ATLAS conversation history or private memory by
  default. A caller may pass selected, approved context as part of a request.
- Each session uses a distinct LangGraph `thread_id`. Do not reuse session ids
  across users or trust boundaries.
- ATHENA has no tools with side effects in this version.

## Execution graph

```text
User request
    |
    v
ANALYSIS (analysis model)
    |
    v
INDEPENDENT CHALLENGE (review model)
    |
    v
SYNTHESIS (analysis model)
    |
    v
Structured response + session checkpoint
```

The reviewer receives the request and analysis, but is prompted to find
counterexamples, unsupported claims, missing evidence, risks, and falsifying
tests. Synthesis must preserve meaningful disagreement and uncertainty.

Analysis and review require different model instances and distinct model
identifiers. For stronger independence, configure different model families or
providers. Different providers can still share training data, incentives, or
failure modes; provider diversity is not proof of correctness.

## Evidence discipline

ATHENA should distinguish:

| Label | Meaning |
|---|---|
| ESTABLISHED | Directly supported by reliable evidence available in the run |
| USER_REPORTED | Supplied by the user, not independently checked |
| INFERENCE | Reasoned from identified premises |
| HYPOTHESIS | Plausible and currently unverified |
| UNKNOWN | Required information is unavailable |

Do not fabricate citations, data, tool execution, experiments, or verification.
Provide concise rationale and an auditable conclusion, not private
chain-of-thought.

## Memory contract

- Memory is scoped by `session_id` / LangGraph `thread_id`.
- The CLI requires an explicit `--session-id` for persistent runs; it has no
  shared default thread.
- Reuse an id only to continue the same logical session. Use a different id for
  each user, project, or trust boundary. A session id is a namespace, not an
  authentication or authorization mechanism.
- Earlier user requests are replayed as `HumanMessage` values and published
  ATHENA answers are replayed as `AIMessage` values. Role fidelity is tested.
- Only the prior user requests and ATHENA's published answers from that same
  thread are included in later analysis context.
- The in-memory checkpointer is used for embedded use, tests, and temporary
  provider smoke runs.
- The CLI uses SQLite checkpoints for persistence between process runs.
- On POSIX, the CLI creates/restricts the checkpoint database file to owner-only
  permissions (`0600`) and creates missing checkpoint directories with owner-only
  permissions (`0700`). Operators remain responsible for existing parent
  directories, backups, retention, and host access. Do not store secrets in prompts.

## Provider support and diagnostics

The initial provider adapters cover OpenAI, Anthropic, and Google Gemini using
the corresponding LangChain integration packages. Provider packages are
optional dependencies so the REOS core remains provider-neutral.

- `reos --doctor` checks required runtime packages, model identifiers,
  recognized provider integration packages, expected credential variable
  presence, and checkpoint storage access. It does not make network calls and
  does not validate the key itself, model entitlement, quota, or connectivity.
- `reos --live-smoke` makes a bounded real model run through all three stages
  using an in-memory checkpointer and a temporary session. Normal provider
  charges may apply.
- Normal task execution uses an explicit `--session-id` and durable SQLite state.

Credential values must never be committed or printed. The GitHub Actions
provider smoke job is manually gated and consumes repository variables for
model identifiers and repository secrets for provider credentials. It is not
triggered by pull-request events.

## Observability

LangSmith tracing is opt-in through environment variables:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT=athena-development`

Graph runs attach `agent`, `agent_version`, `workflow`, and `session_id`
metadata. Do not enable tracing for sensitive inputs unless the project's
privacy, access, and retention settings are appropriate. Trace payloads can
contain prompts and generated outputs.

## Acceptance gates

1. Analysis, review, and synthesis execute in the specified order.
2. Analysis and review use different model instances and distinct model identifiers.
3. A skeptical review is actually invoked and included in synthesis.
4. Prior user/assistant history retains the correct message roles.
5. Separate thread ids do not expose each other's history.
6. Blank requests and blank thread ids are rejected.
7. Unit tests are deterministic and do not require provider credentials.
8. Failed/empty stages cannot be reported as completed turns.
9. CLI session ids are explicit and persistent checkpoint files are owner-only on POSIX.
10. Preflight detects missing runtime dependencies, known provider adapters, and credential variables without exposing secret values.
11. A real provider run is only claimed after the manually gated live smoke test passes on two configured provider/model identifiers.
12. Traces and outputs are only used with appropriate privacy and retention controls.

## Non-goals for this version

- Autonomous access to repositories, email, terminals, cloud resources, or money.
- Autonomous deployment, code merging, or irreversible external actions.
- Automatic self-modification or self-improvement in production.
- A family of specialist agents before ATHENA passes its own provider and output-quality gates.
