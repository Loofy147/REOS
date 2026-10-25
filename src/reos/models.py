from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable
from uuid import uuid4


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"


class ClaimKind(str, Enum):
    FACT = "fact"
    EXPERIMENT = "experiment"
    USER_REPORTED = "user_reported"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"
    CONTRADICTION = "contradiction"
    UNKNOWN = "unknown"
    OPEN_QUESTION = "open_question"


class ArtifactKind(str, Enum):
    FILE = "file"
    LOG = "log"
    PATCH = "patch"
    TEST_RESULT = "test_result"
    REPORT = "report"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Claim:
    id: str
    kind: ClaimKind
    statement: str
    confidence: float = 0.5
    evidence_ids: list[str] = field(default_factory=list)
    source: str = "unknown"
    contradiction_of: str | None = None
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "statement": self.statement,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "source": self.source,
            "contradiction_of": self.contradiction_of,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Claim":
        return cls(
            id=data["id"],
            kind=ClaimKind(data["kind"]),
            statement=data["statement"],
            confidence=float(data.get("confidence", 0.5)),
            evidence_ids=list(data.get("evidence_ids", [])),
            source=data.get("source", "unknown"),
            contradiction_of=data.get("contradiction_of"),
            status=data.get("status", "open"),
        )


@dataclass(slots=True)
class Evidence:
    id: str
    source_type: str
    locator: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "locator": self.locator,
            "summary": self.summary,
            "payload": self.payload,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        return cls(
            id=data["id"],
            source_type=data["source_type"],
            locator=data["locator"],
            summary=data["summary"],
            payload=dict(data.get("payload", {})),
            recorded_at=data.get("recorded_at", utcnow()),
        )


@dataclass(slots=True)
class Artifact:
    id: str
    kind: ArtifactKind
    uri: str
    digest: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "uri": self.uri,
            "digest": self.digest,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            id=data["id"],
            kind=ArtifactKind(data["kind"]),
            uri=data["uri"],
            digest=data.get("digest"),
            description=data.get("description", ""),
        )


@dataclass(slots=True)
class Decision:
    id: str
    rationale: str
    alternatives: list[str] = field(default_factory=list)
    chosen: str = ""
    by_agent: str = "system"
    created_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rationale": self.rationale,
            "alternatives": list(self.alternatives),
            "chosen": self.chosen,
            "by_agent": self.by_agent,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decision":
        return cls(
            id=data["id"],
            rationale=data["rationale"],
            alternatives=list(data.get("alternatives", [])),
            chosen=data.get("chosen", ""),
            by_agent=data.get("by_agent", "system"),
            created_at=data.get("created_at", utcnow()),
        )


@dataclass(slots=True)
class AgentResult:
    role: str
    summary: str
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "summary": self.summary,
            "claims": [claim.to_dict() for claim in self.claims],
            "evidence": [item.to_dict() for item in self.evidence],
            "risks": list(self.risks),
            "next_actions": list(self.next_actions),
            "details": self.details,
        }


@dataclass(slots=True)
class RunRecord:
    id: str
    task_id: str
    stage: str
    agent: str
    status: RunStatus = RunStatus.PENDING
    started_at: str | None = None
    ended_at: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    trace_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "stage": self.stage,
            "agent": self.agent,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "output": self.output,
            "error": self.error,
            "trace_ref": self.trace_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(
            id=data["id"],
            task_id=data["task_id"],
            stage=data["stage"],
            agent=data["agent"],
            status=RunStatus(data.get("status", RunStatus.PENDING.value)),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            output=dict(data.get("output", {})),
            error=data.get("error"),
            trace_ref=data.get("trace_ref"),
        )


@dataclass(slots=True)
class TaskRecord:
    id: str
    objective: str
    scope: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    repository: str | None = None
    branch: str | None = None
    starting_commit: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    assigned_agents: list[str] = field(default_factory=list)
    handoffs: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    rejected_alternatives: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    test_results: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    next_step: str | None = None
    blocked_reason: str | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    runs: list[RunRecord] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = utcnow()

    def add_claim(self, claim: Claim) -> None:
        self.claims.append(claim)
        self.touch()

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)
        self.touch()

    def add_decision(self, decision: Decision) -> None:
        self.decisions.append(decision)
        self.touch()

    def add_run(self, run: RunRecord) -> None:
        self.runs.append(run)
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "scope": self.scope,
            "acceptance_criteria": list(self.acceptance_criteria),
            "repository": self.repository,
            "branch": self.branch,
            "starting_commit": self.starting_commit,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "assigned_agents": list(self.assigned_agents),
            "handoffs": list(self.handoffs),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "assumptions": list(self.assumptions),
            "risks": list(self.risks),
            "rejected_alternatives": list(self.rejected_alternatives),
            "commands": list(self.commands),
            "test_results": list(self.test_results),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "claims": [claim.to_dict() for claim in self.claims],
            "next_step": self.next_step,
            "blocked_reason": self.blocked_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "runs": [run.to_dict() for run in self.runs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRecord":
        return cls(
            id=data["id"],
            objective=data["objective"],
            scope=data.get("scope", ""),
            acceptance_criteria=list(data.get("acceptance_criteria", [])),
            repository=data.get("repository"),
            branch=data.get("branch"),
            starting_commit=data.get("starting_commit"),
            status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
            dependencies=list(data.get("dependencies", [])),
            assigned_agents=list(data.get("assigned_agents", [])),
            handoffs=list(data.get("handoffs", [])),
            decisions=[Decision.from_dict(item) for item in data.get("decisions", [])],
            assumptions=list(data.get("assumptions", [])),
            risks=list(data.get("risks", [])),
            rejected_alternatives=list(data.get("rejected_alternatives", [])),
            commands=list(data.get("commands", [])),
            test_results=list(data.get("test_results", [])),
            artifacts=[Artifact.from_dict(item) for item in data.get("artifacts", [])],
            claims=[Claim.from_dict(item) for item in data.get("claims", [])],
            next_step=data.get("next_step"),
            blocked_reason=data.get("blocked_reason"),
            created_at=data.get("created_at", utcnow()),
            updated_at=data.get("updated_at", utcnow()),
            runs=[RunRecord.from_dict(item) for item in data.get("runs", [])],
        )


@dataclass(slots=True)
class Plan:
    id: str
    task_id: str
    phases: list[dict[str, Any]]
    created_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "task_id": self.task_id, "phases": self.phases, "created_at": self.created_at}


@dataclass(slots=True)
class ExecutionResult:
    task: TaskRecord
    runs: list[RunRecord]
    plan: Plan
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task.to_dict(), "runs": [run.to_dict() for run in self.runs], "plan": self.plan.to_dict(), "notes": list(self.notes)}


@dataclass(slots=True)
class RepositorySnapshot:
    root: str
    branch: str | None = None
    commit: str | None = None
    files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "branch": self.branch,
            "commit": self.commit,
            "files": list(self.files),
            "changed_files": list(self.changed_files),
            "notes": self.notes,
            "metadata": self.metadata,
        }


def coerce_claims(items: Iterable[Claim | dict[str, Any]]) -> list[Claim]:
    claims: list[Claim] = []
    for item in items:
        claims.append(item if isinstance(item, Claim) else Claim.from_dict(item))
    return claims
