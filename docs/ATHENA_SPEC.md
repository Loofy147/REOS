# ATHENA agent contract — v0.1

**Status:** Initial implementation  
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
- ATHENA has no tools with side effects in v0.1.

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

Analysis and review require different model instances. For better independence,
configure different model families or providers. Two instances of one model
with different prompts remain vulnerable to correlated errors.

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
- Only prior user requests and ATHENA's published answers from that same thread
  are included in later analysis context.
- The in-memory checkpointer is a process-local development default.
- The CLI uses SQLite checkpoints for persistence between process runs.
- Treat checkpoint databases as sensitive data: the CLI restricts POSIX database file
  permissions to the invoking OS user, but operators remain responsible for directory
  permissions, backups, retention, and access. Do not store secrets in prompts.

## Observability

LangSmith tracing is opt-in through environment variables:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT=athena-development`

Do not commit API keys or other credentials. Traces may contain prompts and
outputs; apply suitable access and retention controls before using real data.

## Acceptance gates

1. Analysis, review, and synthesis execute in the specified order.
2. Analysis and review use different model instances.
3. A skeptical review is actually invoked and included in synthesis.
4. Separate thread ids do not expose each other's history.
5. Blank requests and blank thread ids are rejected.
6. Unit tests are deterministic and do not require provider credentials.
7. Failed/empty stages cannot be reported as completed turns.
8. CLI session ids are explicit and persistent checkpoint files are owner-only on POSIX.
9. A real provider run is only claimed after separately configured credentials
   and an explicit provider-backed smoke test.

## Non-goals for v0.1

- Autonomous access to repositories, email, terminals, cloud resources, or money.
- Autonomous deployment, code merging, or irreversible external actions.
- Automatic self-modification or self-improvement in production.
- A family of specialist agents before ATHENA passes its own quality gates.
