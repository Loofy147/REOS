from __future__ import annotations

from reos.preflight import build_preflight_report


def test_preflight_fails_when_models_or_provider_credentials_are_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("reos.preflight.importlib.util.find_spec", lambda _name: object())
    report = build_preflight_report(
        analysis_model="openai:model-a",
        review_model="anthropic:model-b",
        database_path=tmp_path / "data" / "athena.sqlite",
        environ={},
    )

    assert report["ok"] is False
    failed = [item["name"] for item in report["checks"] if item["status"] == "FAIL"]
    assert "analysis model credentials" in failed
    assert "review model credentials" in failed


def test_preflight_passes_with_distinct_provider_configs_and_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("reos.preflight.importlib.util.find_spec", lambda _name: object())
    report = build_preflight_report(
        analysis_model="openai:model-a",
        review_model="anthropic:model-b",
        database_path=tmp_path / "data" / "athena.sqlite",
        environ={
            "OPENAI_API_KEY": "do-not-print-openai",
            "ANTHROPIC_API_KEY": "do-not-print-anthropic",
        },
    )

    assert report["ok"] is True
    assert all(item["status"] != "FAIL" for item in report["checks"])
    assert "do-not-print-openai" not in repr(report)
    assert "do-not-print-anthropic" not in repr(report)


def test_preflight_warns_when_both_roles_use_same_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("reos.preflight.importlib.util.find_spec", lambda _name: object())
    report = build_preflight_report(
        analysis_model="openai:model-a",
        review_model="openai:model-b",
        database_path=tmp_path / "data" / "athena.sqlite",
        environ={"OPENAI_API_KEY": "redacted"},
    )

    assert report["ok"] is True
    assert any(
        item["status"] == "WARN" and "same provider" in item["detail"]
        for item in report["checks"]
    )


def test_preflight_rejects_identical_model_specs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("reos.preflight.importlib.util.find_spec", lambda _name: object())
    report = build_preflight_report(
        analysis_model="openai:model-a",
        review_model="openai:model-a",
        database_path=tmp_path / "data" / "athena.sqlite",
        environ={"OPENAI_API_KEY": "redacted"},
    )

    assert report["ok"] is False
    assert any(
        item["name"] == "model separation" and item["status"] == "FAIL"
        for item in report["checks"]
    )
