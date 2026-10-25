from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .agents import (
    DEFAULT_AGENT_SET,
    AgentContext,
    BaseAgent,
)
from .models import (
    AgentResult,
    Artifact,
    ArtifactKind,
    Claim,
    Decision,
    ExecutionResult,
    Plan,
    RepositorySnapshot,
    RunRecord,
    RunStatus,
    TaskRecord,
    TaskStatus,
    new_id,
    utcnow,
)
from .store import JsonStore


class RepositoryGateway(Protocol):
    def snapshot(self) -> RepositorySnapshot:
        ...

    def apply_text_patch(self, path: str, content: str, description: str = "") -> Artifact:
        ...

    def read_text(self, path: str) -> str:
        ...

    def list_files(self) -> list[str]:
        ...


@dataclass(slots=True)
class LocalRepositoryGateway:
    root: Path
    branch: str | None = None
    commit: str | None = None
    notes: str = ""

    def snapshot(self) -> RepositorySnapshot:
        files: list[str] = []
        if self.root.exists():
            for path in self.root.rglob("*"):
                if path.is_file() and ".reos" not in path.parts and ".git" not in path.parts:
                    files.append(str(path.relative_to(self.root)))
        return RepositorySnapshot(
            root=str(self.root),
            branch=self.branch,
            commit=self.commit,
            files=sorted(files),
            notes=self.notes,
            metadata={"kind": "local_filesystem"},
        )

    def apply_text_patch(self, path: str, content: str, description: str = "") -> Artifact:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return Artifact(id=new_id("artifact"), kind=ArtifactKind.FILE, uri=str(target), description=description)

    def read_text(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")

    def list_files(self) -> list[str]:
        return self.snapshot().files


@dataclass(slots=True)
class ExecutionContext:
    task: TaskRecord
    repository_snapshot: dict[str, Any]
    constraints: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)


class ReosEngine:
    def __init__(self, store: JsonStore | None = None, agents: dict[str, BaseAgent] | None = None) -> None:
        self.store = store or JsonStore()
        self.agents = agents or DEFAULT_AGENT_SET

    def create_task(
        self,
        objective: str,
        repository: str | None = None,
        scope: str = "",
        acceptance_criteria: list[str] | None = None,
        dependencies: list[str] | None = None,
    ) -> TaskRecord:
        task = TaskRecord(
            id=new_id("task"),
            objective=objective,
            scope=scope,
            repository=repository,
            acceptance_criteria=acceptance_criteria or [],
            dependencies=dependencies or [],
        )
        self.store.save_task(task)
        return task

    def select_team(self, task: TaskRecord) -> list[str]:
        objective = task.objective.lower()
        team = ["requirements_architecture", "security", "independent_reviewer"]
        if task.repository:
            team.insert(0, "repository_scout")
        if any(token in objective for token in ("research", "discover", "inspect", "investigate")):
            team.append("research")
        if any(token in objective for token in ("implement", "build", "change", "fix", "modify", "update")):
            team.append("implementation")
        if any(token in objective for token in ("test", "verify", "validate", "regression", "quality")):
            team.append("test_reliability")
        team.append("integration_verification")
        # preserve order while deduplicating
        seen: set[str] = set()
        ordered: list[str] = []
        for role in team:
            if role not in seen and role in self.agents:
                ordered.append(role)
                seen.add(role)
        return ordered

    def build_plan(self, task: TaskRecord) -> Plan:
        team = self.select_team(task)
        task.assigned_agents = team
        task.acceptance_criteria = task.acceptance_criteria or [
            "Objective decomposed into structured tasks and evidence records.",
            "Relevant agents selected and delegated in a deterministic order.",
            "Failures, risks, and unknowns are persisted rather than discarded.",
            "Final result is reviewable and resumable.",
        ]
        phases = [
            {"name": "discover", "agents": ["repository_scout"] if "repository_scout" in team else []},
            {"name": "research", "agents": ["research"] if "research" in team else []},
            {"name": "plan", "agents": ["requirements_architecture", "security"]},
            {"name": "implement", "agents": ["implementation"] if "implementation" in team else []},
            {"name": "test", "agents": ["test_reliability"] if "test_reliability" in team else []},
            {"name": "review", "agents": ["independent_reviewer"]},
            {"name": "verify", "agents": ["integration_verification"]},
            {"name": "record", "agents": []},
        ]
        plan = Plan(id=new_id("plan"), task_id=task.id, phases=phases)
        task.next_step = phases[0]["name"]
        task.touch()
        self.store.save_task(task)
        return plan

    def _agent_context(self, task: TaskRecord, snapshot: RepositorySnapshot, evidence: list[dict[str, Any]]) -> AgentContext:
        return AgentContext(
            objective=task.objective,
            repository_snapshot=snapshot.to_dict(),
            task=task.to_dict(),
            evidence=evidence,
            constraints={"acceptance_criteria": task.acceptance_criteria},
        )

    def _record_agent_result(self, task: TaskRecord, stage: str, result: AgentResult) -> RunRecord:
        run = RunRecord(
            id=new_id("run"),
            task_id=task.id,
            stage=stage,
            agent=result.role,
            status=RunStatus.SUCCEEDED,
            started_at=utcnow(),
            ended_at=utcnow(),
            output=result.to_dict(),
        )
        task.add_run(run)
        for item in result.claims:
            task.add_claim(item)
        for item in result.evidence:
            task.add_artifact(
                Artifact(
                    id=item.id,
                    kind=ArtifactKind.LOG,
                    uri=item.locator,
                    description=item.summary,
                )
            )
            self.store.append_evidence(item.to_dict())
        task.risks.extend(result.risks)
        task.handoffs.append({"stage": stage, "agent": result.role, "summary": result.summary, "next_actions": result.next_actions})
        task.touch()
        self.store.save_run(run)
        self.store.save_task(task)
        return run

    def _run_agent(self, task: TaskRecord, stage: str, snapshot: RepositorySnapshot, evidence: list[dict[str, Any]]) -> RunRecord:
        agent = self.agents[stage]
        run = RunRecord(id=new_id("run"), task_id=task.id, stage=stage, agent=agent.role, status=RunStatus.RUNNING, started_at=utcnow())
        task.add_run(run)
        self.store.save_run(run)
        try:
            result = agent.run(self._agent_context(task, snapshot, evidence))
            run.status = RunStatus.SUCCEEDED
            run.ended_at = utcnow()
            run.output = result.to_dict()
            self._record_agent_result(task, stage, result)
        except Exception as exc:  # pragma: no cover - defensive boundary
            run.status = RunStatus.FAILED
            run.error = f"{type(exc).__name__}: {exc}"
            run.ended_at = utcnow()
            task.status = TaskStatus.FAILED
            task.blocked_reason = run.error
            task.next_step = stage
            task.touch()
            self.store.save_run(run)
            self.store.save_task(task)
        return run

    def execute(
        self,
        task: TaskRecord | str,
        repository_gateway: RepositoryGateway | None = None,
        allow_write: bool = False,
    ) -> ExecutionResult:
        task_record = task if isinstance(task, TaskRecord) else self.store.load_task(task)
        plan = self.build_plan(task_record)
        task_record.status = TaskStatus.RUNNING
        task_record.touch()
        self.store.save_task(task_record)
        snapshot = repository_gateway.snapshot() if repository_gateway else RepositorySnapshot(root=task_record.repository or "unknown")
        evidence: list[dict[str, Any]] = []
        runs: list[RunRecord] = []

        parallel_stages = [stage for stage in ("repository_scout", "research", "requirements_architecture", "security") if stage in plan.phases[0]["agents"] + plan.phases[1]["agents"] + plan.phases[2]["agents"]]
        with ThreadPoolExecutor(max_workers=max(1, len(parallel_stages))) as pool:
            future_map = {}
            for stage in parallel_stages:
                if stage in self.agents:
                    future_map[pool.submit(self._run_agent, task_record, stage, snapshot, evidence)] = stage
            for future in as_completed(future_map):
                runs.append(future.result())

        evidence.extend([claim.to_dict() for claim in task_record.claims])

        sequential_stages = [
            stage
            for stage in ("implementation", "test_reliability", "independent_reviewer", "integration_verification")
            if stage in self.agents and stage in task_record.assigned_agents
        ]
        for stage in sequential_stages:
            if stage == "implementation" and not allow_write:
                task_record.status = TaskStatus.BLOCKED
                task_record.blocked_reason = "Write-capable execution not authorized in this run."
                task_record.next_step = stage
                task_record.touch()
                self.store.save_task(task_record)
                break
            runs.append(self._run_agent(task_record, stage, snapshot, evidence))
            evidence = [claim.to_dict() for claim in task_record.claims]

        if task_record.status not in (TaskStatus.BLOCKED, TaskStatus.FAILED):
            task_record.status = TaskStatus.COMPLETED
            task_record.next_step = None
            task_record.blocked_reason = None
            task_record.add_decision(
                Decision(
                    id=new_id("decision"),
                    rationale="Execution completed with persisted task, run, claim, and evidence records.",
                    alternatives=["block on missing write authorization", "stop at planning only"],
                    chosen="complete",
                    by_agent="engine",
                )
            )
            task_record.touch()
            self.store.save_task(task_record)
        result = ExecutionResult(task=task_record, runs=runs, plan=plan, notes=["LangGraph and LangSmith integrations are optional and adapter-based."])
        self.store.save_execution_result(result)
        return result

    def resume(self, task_id: str, repository_gateway: RepositoryGateway | None = None, allow_write: bool = False) -> ExecutionResult:
        return self.execute(task_id, repository_gateway=repository_gateway, allow_write=allow_write)
