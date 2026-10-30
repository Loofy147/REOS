# REOS — Reusable Engineering Orchestration System

REOS is a provider-neutral foundation for independently configured AI agents and
coordinated, evidence-oriented engineering workflows.

The first reference agent is **ATHENA**: an independent strategic reasoning
agent with a separate analysis model, a separate skeptical reviewer, thread-scoped
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
- The core remains provider-neutral. OpenAI, Anthropic, and Google Gemini
  adapters are optional integrations.

## Quick start

Install the runtime plus the providers you intend to use:

```bash
python -m pip install -e ".[lang,openai,anthropic,dev]"
```

Set model identifiers and credentials in the environment, then check readiness:

```bash
export ATHENA_ANALYSIS_MODEL="openai:<analysis-model>"
export ATHENA_REVIEW_MODEL="anthropic:<review-model>"
export OPENAI_API_KEY="<your-openai-key>"
export ANTHROPIC_API_KEY="<your-anthropic-key>"
reos --doctor
reos --live-smoke
```

Run a persistent task by supplying a unique session namespace:

```bash
reos "Assess the next engineering priority and identify the falsifying test." \
  --session-id portfolio-001 \
  --database ./data/athena.sqlite
```

See [ATHENA's specification](docs/ATHENA_SPEC.md) for its contract and
[getting started](docs/GETTING_STARTED.md) for provider setup, safe credential
handling, and the manually gated GitHub Actions smoke workflow.
