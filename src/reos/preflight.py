from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Mapping


_PROVIDER_PROFILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "openai": ("langchain_openai", ("OPENAI_API_KEY",)),
    "anthropic": ("langchain_anthropic", ("ANTHROPIC_API_KEY",)),
    "google_genai": ("langchain_google_genai", ("GOOGLE_API_KEY", "GEMINI_API_KEY")),
}


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _model_provider(model_spec: str) -> tuple[str, str] | None:
    if not isinstance(model_spec, str) or ":" not in model_spec:
        return None
    provider, model_name = model_spec.split(":", 1)
    provider, model_name = provider.strip(), model_name.strip()
    if not provider or not model_name:
        return None
    return provider, model_name


def build_preflight_report(
    *,
    analysis_model: str,
    review_model: str,
    database_path: str | Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Inspect local runtime readiness without printing or transmitting credentials."""
    env = os.environ if environ is None else environ
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    for module_name, label in (
        ("langchain", "LangChain runtime"),
        ("langgraph", "LangGraph runtime"),
        ("langgraph.checkpoint.sqlite", "SQLite checkpoint runtime"),
    ):
        available = _module_available(module_name)
        add(
            label,
            "PASS" if available else "FAIL",
            "installed" if available else f"missing; install REOS with the 'lang' extra",
        )

    parsed_models: list[tuple[str, str] | None] = []
    for role, spec in (("analysis", analysis_model), ("review", review_model)):
        parsed = _model_provider(spec)
        parsed_models.append(parsed)
        if parsed is None:
            add(
                f"{role} model identifier",
                "FAIL",
                "missing or invalid; expected a provider-prefixed value such as openai:model-name",
            )
            continue

        provider, _model_name = parsed
        profile = _PROVIDER_PROFILES.get(provider)
        if profile is None:
            add(
                f"{role} provider integration",
                "WARN",
                f"provider '{provider}' is not in the built-in preflight map; verify its LangChain integration and credentials manually",
            )
            continue

        module_name, credential_names = profile
        integration_available = _module_available(module_name)
        add(
            f"{role} provider integration",
            "PASS" if integration_available else "FAIL",
            "installed" if integration_available else f"missing; install the '{provider}' extra",
        )

        if credential_names:
            present = [name for name in credential_names if env.get(name, "").strip()]
            add(
                f"{role} model credentials",
                "PASS" if present else "FAIL",
                f"configured via {', '.join(present)}" if present else f"missing; set one of: {', '.join(credential_names)}",
            )

    if parsed_models[0] is None or parsed_models[1] is None:
        add("model separation", "FAIL", "both roles need valid provider-prefixed model identifiers")
    elif analysis_model.strip() == review_model.strip():
        add("model separation", "FAIL", "analysis and review model identifiers are identical")
    elif parsed_models[0][0] == parsed_models[1][0]:
        add(
            "provider independence",
            "WARN",
            "both roles use the same provider; prefer different model families or providers to reduce correlated error",
        )
    else:
        add(
            "provider diversity",
            "INFO",
            "analysis and review use different providers; this may reduce shared provider-specific failures but does not establish statistical independence or correctness",
        )

    db_path = Path(database_path).expanduser()
    if db_path.exists():
        writable = os.access(db_path, os.R_OK | os.W_OK)
        add(
            "checkpoint database",
            "PASS" if writable else "FAIL",
            "existing database is readable and writable" if writable else "existing database is not readable and writable",
        )
    else:
        parent = db_path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        writable = parent.exists() and os.access(parent, os.W_OK)
        add(
            "checkpoint directory",
            "PASS" if writable else "FAIL",
            f"nearest existing parent is writable: {parent}" if writable else f"no writable parent directory found for {db_path}",
        )

    tracing = env.get("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes"}
    trace_key = bool(env.get("LANGSMITH_API_KEY", "").strip())
    if tracing and trace_key:
        add("LangSmith tracing", "PASS", "enabled with a configured API key; review trace privacy before real data")
    elif tracing:
        add("LangSmith tracing", "WARN", "tracing is enabled but LANGSMITH_API_KEY is not configured")
    else:
        add("LangSmith tracing", "INFO", "disabled; this is valid for local execution")

    return {
        "ok": not any(item["status"] == "FAIL" for item in checks),
        "checks": checks,
    }
