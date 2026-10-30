# ATHENA AppDeploy-hosted runtime

**Status:** deployed, ready; first authenticated LLM turn not yet independently verified  
**App:** ATHENA  
**App ID:** `athena-t60myz`  
**Live URL:** https://athena-t60myz.v2.appdeploy.ai/  
**Source snapshot:** `1793341303260`

## Why this runtime exists

AppDeploy exposes its hosted model service to application backends through
`ai.generate()` in `@appdeploy/sdk`. This avoids requiring separate OpenAI,
Anthropic, or Gemini API keys for the hosted app.

The Python REOS package cannot import the AppDeploy JavaScript SDK directly.
The deployed AppDeploy app therefore provides a native hosted runtime and UI.
It implements the same three-stage reasoning contract, but it is currently a
companion runtime rather than a Python-to-AppDeploy remote provider adapter.

## Execution contract

An authenticated user submits one question. The backend executes three
sequential `ai.generate()` calls:

1. **Analysis:** identify objectives, constraints, evidence, assumptions,
   options, risks, and discriminating tests.
2. **Skeptical review:** actively challenge the analysis and identify
   counterexamples, unsupported assumptions, and falsifying tests.
3. **Synthesis:** preserve disagreements and uncertainty in a decision brief.

Each stage has a 900-token output cap. The input question is limited to 5,000
characters. A session is capped at 10 completed turns. Up to four prior
question/answer pairs are included as context in the next analysis.

## Independence limitation

The AppDeploy `ai.generate` options returned by the SDK do not expose a
per-call model selector. The three roles are separate prompts and separate
generation calls through the AppDeploy-hosted model service. This is a
multi-pass review process, **not** independent model-family verification.

Do not label the reviewer as statistically independent. Genuine model
diversity remains a separate future gate requiring an interface that can select
distinct model configurations or providers.

## Authentication and persistence

- AppDeploy `auth.getUser()` restores the signed-in user in the frontend.
- Every session endpoint except the health check uses AppDeploy `requireAuth()`.
- Session tables are partitioned by the authenticated AppDeploy user ID.
- Sessions and completed turns persist in the AppDeploy database.
- Session updates use version-conditional database patches to reject stale
  writes rather than silently overwrite a concurrent update.
- The app never places provider API keys in frontend code or source files.
- The three generation stages are called only after a signed-in user submits a
  question.

## Routes

| Route | Purpose |
|---|---|
| `GET /api/_healthcheck` | Public service health check; no model call |
| `GET /api/sessions` | List the authenticated user's sessions |
| `POST /api/sessions` | Create a private session |
| `GET /api/sessions/:id` | Read one session from the user's partition |
| `POST /api/sessions/:id/turn` | Execute analysis, challenge, synthesis, then persist |
| `DELETE /api/sessions/:id` | Delete a session after UI confirmation |

## Verification record

- AppDeploy deployment state: `ready`.
- AppDeploy QA snapshot: zero reported frontend errors, zero network errors.
- AppDeploy backend error list: empty at the time of status inspection.
- Four user-visible acceptance flows are registered in `tests/tests.json`:
  primary three-pass run + reload, empty-input rejection, confirmed deletion,
  and mobile navigation/composer.
- **Not yet established:** a signed-in request has completed all three real
  model calls and persisted the resulting turn. AppDeploy status checks and
  screenshot capture do not alone prove that an authenticated AI turn worked.

## Quality-gate parity status

The deterministic quality gate described in `docs/ATHENA_SPEC.md` currently exists in the REOS Python runtime only. It has **not** been implemented in this AppDeploy-hosted snapshot. The deployed app must not be described as having numeric provenance checks, registered-evidence enforcement, sensitive-data recommendation blocking, decision-consistency checks, or quality-report output until a separate AppDeploy source update is deployed and its authenticated runtime is verified.

## REOS relationship and next gate

The Python REOS agent and this AppDeploy-hosted runtime are two implementations
of the ATHENA contract at this stage. No Python HTTP adapter or authenticated
service-to-service bridge has been wired between them. The user-facing hosted
runtime is usable after AppDeploy sign-in; the next discriminating test is one
real authenticated question, followed by reload and retrieval of its saved
result. Then evaluate output quality on a small fixed corpus before adding
specialist agents or claiming independent verification.
