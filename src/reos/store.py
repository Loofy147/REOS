from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable

from .models import ExecutionResult, RunRecord, TaskRecord


class JsonStore:
    def __init__(self, root: str | Path = ".reos") -> None:
        self.root = Path(root)
        self.tasks_dir = self.root / "tasks"
        self.runs_dir = self.root / "runs"
        self.artifacts_dir = self.root / "artifacts"
        self.evidence_dir = self.root / "evidence"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    def save_task(self, task: TaskRecord) -> Path:
        path = self.tasks_dir / f"{task.id}.json"
        self._atomic_write(path, task.to_dict())
        return path

    def load_task(self, task_id: str) -> TaskRecord:
        path = self.tasks_dir / f"{task_id}.json"
        return TaskRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_tasks(self) -> list[TaskRecord]:
        tasks: list[TaskRecord] = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            tasks.append(TaskRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return tasks

    def save_run(self, run: RunRecord) -> Path:
        path = self.runs_dir / f"{run.id}.json"
        self._atomic_write(path, run.to_dict())
        return path

    def load_run(self, run_id: str) -> RunRecord:
        path = self.runs_dir / f"{run_id}.json"
        return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_execution_result(self, result: ExecutionResult) -> Path:
        path = self.root / f"{result.task.id}_result.json"
        self._atomic_write(path, result.to_dict())
        return path

    def append_evidence(self, evidence: dict) -> Path:
        path = self.evidence_dir / f"{evidence['id']}.json"
        self._atomic_write(path, evidence)
        return path

    def task_artifact_path(self, task_id: str, name: str) -> Path:
        path = self.artifacts_dir / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path / name

    def existing_run_stages(self, task: TaskRecord) -> set[str]:
        return {run.stage for run in task.runs if run.status.value == "succeeded"}

    def recent_runs(self, task_id: str) -> list[RunRecord]:
        runs: list[RunRecord] = []
        for path in sorted(self.runs_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("task_id") == task_id:
                runs.append(RunRecord.from_dict(data))
        return runs

    def records(self) -> dict[str, list[str]]:
        return {
            "tasks": [path.name for path in self.tasks_dir.glob("*.json")],
            "runs": [path.name for path in self.runs_dir.glob("*.json")],
            "evidence": [path.name for path in self.evidence_dir.glob("*.json")],
        }
