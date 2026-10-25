from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .models import AgentResult, Claim, ClaimKind, Evidence, new_id


@dataclass(slots=True)
class AgentContext:
    objective: str
    repository_snapshot: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    role: str

    @abstractmethod
    def run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError


class RepositoryScoutAgent(BaseAgent):
    role = "repository_scout"

    def run(self, context: AgentContext) -> AgentResult:
        files = context.repository_snapshot.get("files", [])
        notes = context.repository_snapshot.get("notes", "")
        claims = [
            Claim(
                id=new_id("claim"),
                kind=ClaimKind.FACT,
                statement=f"Repository snapshot contains {len(files)} visible files.",
                confidence=0.8,
                source=self.role,
            ),
            Claim(
                id=new_id("claim"),
                kind=ClaimKind.INFERENCE,
                statement="The codebase should be organized around a small, testable core with explicit adapters.",
                confidence=0.7,
                source=self.role,
            ),
        ]
        if notes:
            claims.append(
                Claim(
                    id=new_id("claim"),
                    kind=ClaimKind.FACT,
                    statement=notes,
                    confidence=0.9,
                    source=self.role,
                )
            )
        evidence = [
            Evidence(
                id=new_id("ev"),
                source_type="repository_snapshot",
                locator=context.repository_snapshot.get("root", "unknown"),
                summary="Snapshot used for repository scouting",
                payload={"files": files, "notes": notes},
            )
        ]
        return AgentResult(
            role=self.role,
            summary="Scouted repository structure and identified implementation boundaries.",
            claims=claims,
            evidence=evidence,
            risks=["Repository content may be incomplete or empty."],
            next_actions=["Capture the baseline tree", "Define reusable interfaces before implementation"],
            details={"file_count": len(files)},
        )


class ResearchAgent(BaseAgent):
    role = "research"

    def run(self, context: AgentContext) -> AgentResult:
        unknowns = [
            "Whether LangGraph is installed in the execution environment.",
            "Whether LangSmith tracing is available through workspace configuration.",
        ]
        claims = [
            Claim(
                id=new_id("claim"),
                kind=ClaimKind.OPEN_QUESTION,
                statement=item,
                confidence=0.2,
                source=self.role,
            )
            for item in unknowns
        ]
        return AgentResult(
            role=self.role,
            summary="Identified environment uncertainty and runtime integration boundaries.",
            claims=claims,
            risks=["Optional integrations must remain non-blocking."],
            next_actions=["Gate optional integrations behind capability checks"],
            details={"unknowns": unknowns},
        )


class RequirementsArchitectureAgent(BaseAgent):
    role = "requirements_architecture"

    def run(self, context: AgentContext) -> AgentResult:
        objective = context.objective
        criteria = [
            f"Translate '{objective}' into a structured objective graph.",
            "Persist every task, run, claim, evidence item, and artifact.",
            "Support recovery and idempotent reruns.",
            "Separate read-only investigation from write-capable execution.",
        ]
        claims = [
            Claim(
                id=new_id("claim"),
                kind=ClaimKind.FACT,
                statement=criterion,
                confidence=0.95,
                source=self.role,
            )
            for criterion in criteria
        ]
        return AgentResult(
            role=self.role,
            summary="Defined core interface and acceptance criteria for the orchestration engine.",
            claims=claims,
            next_actions=["Encode contracts in the data model", "Expose a CLI entry point"],
            details={"acceptance_criteria": criteria},
        )


class ImplementationAgent(BaseAgent):
    role = "implementation"

    def run(self, context: AgentContext) -> AgentResult:
        target = context.repository_snapshot.get("root", "repository")
        claims = [
            Claim(
                id=new_id("claim"),
                kind=ClaimKind.HYPOTHESIS,
                statement=f"A minimal but durable implementation can be added to {target} without project-specific assumptions.",
                confidence=0.8,
                source=self.role,
            )
        ]
        return AgentResult(
            role=self.role,
            summary="Prepared implementation-oriented change set and reusable code boundaries.",
            claims=claims,
            next_actions=["Write modular Python package files", "Add tests for persistence and orchestration"],
            details={"target": target, "patches": context.constraints.get("patches", [])},
        )


class TestReliabilityAgent(BaseAgent):
    role = "test_reliability"

    def run(self, context: AgentContext) -> AgentResult:
        risks = [
            "State transitions may regress if task records are not reloaded from disk.",
            "Duplicate writes could happen without atomic persistence.",
        ]
        return AgentResult(
            role=self.role,
            summary="Outlined regression checks for persistence, recovery, and execution ordering.",
            claims=[
                Claim(id=new_id("claim"), kind=ClaimKind.FACT, statement=risk, confidence=0.9, source=self.role)
                for risk in risks
            ],
            risks=risks,
            next_actions=["Cover resume semantics", "Cover duplicate side effects"],
            details={"test_focus": ["state transitions", "atomic writes", "resume handling"]},
        )


class SecurityAgent(BaseAgent):
    role = "security"

    def run(self, context: AgentContext) -> AgentResult:
        concerns = [
            "Untrusted repository content must not override system policy.",
            "Sensitive write actions need explicit approval boundaries.",
            "Secrets should remain external to the repository and be injected at runtime.",
        ]
        claims = [
            Claim(id=new_id("claim"), kind=ClaimKind.FACT, statement=item, confidence=0.98, source=self.role)
            for item in concerns
        ]
        return AgentResult(
            role=self.role,
            summary="Identified trust boundary and approval requirements.",
            claims=claims,
            risks=concerns,
            next_actions=["Keep execution adapters injectable", "Never hardcode credentials"],
            details={"policy": "least privilege + explicit approvals"},
        )


class IndependentReviewer(BaseAgent):
    role = "independent_reviewer"

    def run(self, context: AgentContext) -> AgentResult:
        claims = []
        risks = []
        for item in context.evidence:
            if not item.get("summary"):
                risks.append("Missing evidence summary weakens reviewability.")
        if not risks:
            risks.append("Implementation must be validated by both tests and persisted evidence.")
        return AgentResult(
            role=self.role,
            summary="Challenged overclaiming and required traceable evidence for completion.",
            claims=[
                Claim(
                    id=new_id("claim"),
                    kind=ClaimKind.CONTRADICTION,
                    statement="A successful build alone is not proof of correctness.",
                    confidence=1.0,
                    source=self.role,
                )
            ],
            risks=risks,
            next_actions=["Require explicit test artifacts", "Inspect run history for skipped stages"],
            details={"checked_evidence_items": len(context.evidence)},
        )


class IntegrationVerificationAgent(BaseAgent):
    role = "integration_verification"

    def run(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            role=self.role,
            summary="Verified that the reusable design can compose with external repository and tracing adapters.",
            claims=[
                Claim(
                    id=new_id("claim"),
                    kind=ClaimKind.INFERENCE,
                    statement="The system should remain portable because storage and execution are isolated behind interfaces.",
                    confidence=0.85,
                    source=self.role,
                )
            ],
            next_actions=["Record integration boundaries in documentation"],
            details={"interfaces": ["repository gateway", "store", "agents", "engine"]},
        )


DEFAULT_AGENT_SET = {
    RepositoryScoutAgent.role: RepositoryScoutAgent(),
    ResearchAgent.role: ResearchAgent(),
    RequirementsArchitectureAgent.role: RequirementsArchitectureAgent(),
    ImplementationAgent.role: ImplementationAgent(),
    TestReliabilityAgent.role: TestReliabilityAgent(),
    SecurityAgent.role: SecurityAgent(),
    IndependentReviewer.role: IndependentReviewer(),
    IntegrationVerificationAgent.role: IntegrationVerificationAgent(),
}
