# Getting started with ATHENA

## 1. Requirements

- Python 3.10+
- A virtual environment
- Credentials for the two configured model providers

## 2. Install the runtime and provider integrations

Core runtime and tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[lang,dev]"
```

Choose the integrations required by your analysis/review model pair:

```bash
# OpenAI + Anthropic (recommended cross-provider starting point)
python -m pip install -e ".[lang,openai,anthropic,dev]"

# OpenAI + Google Gemini
python -m pip install -e ".[lang,openai,google-genai,dev]"

# All three supported integrations
python -m pip install -e ".[lang,openai,anthropic,google-genai,dev]"
```

## 3. Configure model identities and credentials

Use two distinct provider-prefixed model identifiers. The model names below
are placeholders; choose model IDs supported by your provider account.

```bash
export ATHENA_ANALYSIS_MODEL="openai:<analysis-model>"
export ATHENA_REVIEW_MODEL="anthropic:<review-model>"
export OPENAI_API_KEY="<your-openai-key>"
export ANTHROPIC_API_KEY="<your-anthropic-key>"
```

For Gemini, use `google_genai:<model-name>` and set either
`GOOGLE_API_KEY` or `GEMINI_API_KEY`.

Different model families or providers reduce some shared failure modes, but
do not guarantee statistical independence or correctness. Two models from the
same provider can still have correlated weaknesses.

## 4. Check readiness before spending on model calls

```bash
reos --doctor
```

The doctor checks the LangChain/LangGraph runtime, provider integration
packages, presence of expected credential environment variables, and local
checkpoint storage permissions. It never prints credential values and does not
make provider requests. A passing doctor check does not validate that a key is
accepted, that a model is available to your account, or that quota/network
access is working.

## 5. Run a provider-backed smoke test

```bash
reos --live-smoke
```

This makes a small real run through analysis → independent challenge →
synthesis. It uses an in-memory checkpoint and a temporary session ID, so it
does not modify the persistent ATHENA session database. The run uses two model
calls on the analysis model and one on the review model; normal provider charges
may apply.

## 6. Run a real task with persistent session state

```bash
reos "Assess this product launch plan and identify the next validation test." \
  --session-id product-launch-01 \
  --database ./data/athena.sqlite
```

Reuse the same `--session-id` only to continue that session. Choose a fresh id
for each user, project, or trust boundary. A session id is a namespace, not an
authentication mechanism. The SQLite file is persistent, sensitive application
data. On POSIX the CLI creates/restricts it to owner-only permissions (0600);
keep its parent directory, backups, and host access similarly protected.

## 7. Run the provider smoke test through GitHub Actions

The REOS verification workflow can execute the smoke test on the ATHENA feature
branch when both model identifiers are configured in repository variables.

In GitHub, open **Settings → Secrets and variables → Actions** and add these
repository variables:

- `ATHENA_ANALYSIS_MODEL` — for example, `openai:<model-id>`
- `ATHENA_REVIEW_MODEL` — for example, `anthropic:<model-id>`

Add provider credentials as repository **Secrets**, never as variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` when using Gemini

The workflow runs `--doctor` first. Missing expected credentials stop the
job before model calls. When the variables and secrets are present, the next
push to `feat/athena-independent-agent-v0.1` automatically triggers the
provider-backed smoke test. A manually dispatched smoke is also available
after this workflow is present on the repository's default branch.

The smoke prompt is synthetic and bounded, but valid credentials may incur
provider charges. Do not place private project or user data in this test.

## 8. Enable LangSmith tracing (optional)

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="<your-langsmith-key>"
export LANGSMITH_PROJECT="athena-development"
```

ATHENA attaches `agent`, `agent_version`, `workflow`, and `session_id`
metadata to each graph run. Traces can contain prompts and generated outputs.
Do not enable tracing for sensitive inputs unless the project's privacy,
access, and retention settings are appropriate.

## 9. Verify the local deterministic contract

```bash
python -m pytest -q
```

The acceptance tests use deterministic local model doubles. A passing unit
suite proves graph behavior, correct conversation roles, failure handling,
and session isolation; it does not prove a real provider credential, provider
quota, or production deployment works.
