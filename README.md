# REOS — Reusable Engineering Orchestration System

REOS is a provider-neutral foundation for independently configured AI agents and
coordinated, evidence-oriented engineering workflows.

The first reference agent is **ATHENA**: an independent strategic reasoning
agent with a separate analysis model, a separate reviewer model, thread-scoped
memory, and explicit LangSmith observability support.

## Design constraints

- Agent identity, model configuration, memory, and permissions are separate concerns.
- ATHENA does not inherit ATLAS's private session memory. Context is shared only
  when explicitly supplied to a run.
- Analysis and review use distinct model instances. Use different model
  families/providers when genuine model diversity is required; separate prompts
  alone do not guarantee independent judgments.
- Claims must distinguish evidence from inference and unknowns.
- No external side effect is executed without an explicitly granted tool and
  permission policy.
- The core remains provider-neutral. Provider integrations are optional.

## Status

ATHENA v0.1 is the first implementation slice. It establishes a testable
analysis → independent challenge → synthesis graph and session-isolated memory.
A real provider-backed run requires model credentials and provider integration;
tests use deterministic local model doubles and never call external APIs.

See [ATHENA's specification](docs/ATHENA_SPEC.md) for the contract and
[getting started](docs/GETTING_STARTED.md) for setup.
