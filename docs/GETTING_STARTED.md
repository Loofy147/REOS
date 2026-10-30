# Getting started with ATHENA

## 1. Requirements

- Python 3.10+
- A virtual environment
- A LangChain chat-model provider integration for the model you intend to use

## 2. Install

Core package and tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[lang,dev]"
```

For the OpenAI provider, also install the optional integration:

```bash
python -m pip install -e ".[lang,openai,dev]"
```

## 3. Configure model identities

Set two distinct provider-prefixed model identifiers. The identifiers below
are examples; select models and provider integrations available to your account.

```bash
export ATHENA_ANALYSIS_MODEL="openai:<analysis-model>"
export ATHENA_REVIEW_MODEL="openai:<review-model>"
export OPENAI_API_KEY="<your-key>"
```

Use different model families/providers when stronger independence is required.
Different prompt templates on the same model do not eliminate correlated error.

## 4. Run ATHENA

```bash
reos "Assess this product launch plan and identify the next validation test." \
  --session-id product-launch-01 \
  --database ./data/athena.sqlite
```

Reuse the same `--session-id` only to continue that session. The argument is
required so separate CLI runs cannot silently share a generic default thread.
Choose a fresh id for each user, project, or trust boundary. Do not use a
session id as an authentication mechanism. The SQLite file is persistent, sensitive application data. On POSIX the CLI
creates/restricts it to owner-only permissions (0600); keep its parent directory,
backups, and host access similarly protected.

## 5. Enable LangSmith tracing (optional)

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="<your-langsmith-key>"
export LANGSMITH_PROJECT="athena-development"
```

Traces can contain prompts and generated outputs. Do not enable tracing for
sensitive inputs unless the project’s privacy, access, and retention settings
are appropriate.

## 6. Verify

```bash
python -m pytest -q
```

The acceptance tests use local deterministic model doubles. A passing unit
suite proves graph behavior and session isolation; it does not prove a real
provider credential, provider quota, or production deployment works.
