"""In-process A2A message bus for agent communication.

Routes tasks between agents within the same process, maintaining
a full audit log of all inter-agent communication.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .protocol import AgentCard, Artifact, Message, Part, Task, TaskState


class A2ABus:
    """In-process A2A message bus — routes tasks between agents."""

    def __init__(self):
        self._agents: dict[str, AgentCard] = {}
        self._task_store: dict[str, Task] = {}

    def register_agent(self, card: AgentCard) -> None:
        self._agents[card.agent_id] = card

    def send_task(
        self,
        sender_id: str,
        receiver_id: str,
        task_type: str,
        message_parts: list[Part] | None = None,
    ) -> Task:
        task_id = uuid.uuid4().hex[:12]
        now = time.time()

        msg = Message(
            role="user",
            sender_id=sender_id,
            parts=message_parts or [],
            timestamp=now,
        )

        task = Task(
            task_id=task_id,
            task_type=task_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            state=TaskState.IN_PROGRESS,
            messages=[msg],
            artifacts=[],
            created_at=now,
        )
        self._task_store[task_id] = task
        return task

    def complete_task(
        self,
        task_id: str,
        artifacts: list[Artifact] | None = None,
        response_parts: list[Part] | None = None,
    ) -> Task:
        task = self._task_store[task_id]
        task.state = TaskState.COMPLETED
        task.completed_at = time.time()

        if artifacts:
            task.artifacts.extend(artifacts)

        if response_parts:
            msg = Message(
                role="agent",
                sender_id=task.receiver_id,
                parts=response_parts,
                timestamp=task.completed_at,
            )
            task.messages.append(msg)

        return task

    def fail_task(self, task_id: str, error: str) -> Task:
        task = self._task_store[task_id]
        task.state = TaskState.FAILED
        task.completed_at = time.time()
        task.messages.append(Message(
            role="agent",
            sender_id=task.receiver_id,
            parts=[Part(type="text", content=f"ERROR: {error}")],
            timestamp=task.completed_at,
        ))
        return task

    def get_task(self, task_id: str) -> Task:
        return self._task_store[task_id]

    def get_history(self) -> list[Task]:
        return list(self._task_store.values())

    def summary(self) -> dict:
        from collections import defaultdict
        stats: dict[str, Any] = defaultdict(lambda: {"count": 0, "total_time": 0.0})
        for task in self._task_store.values():
            s = stats[task.task_type]
            s["count"] += 1
            if task.completed_at and task.created_at:
                s["total_time"] += task.completed_at - task.created_at
        result = {}
        for tt, s in stats.items():
            result[tt] = {
                "count": s["count"],
                "avg_time": s["total_time"] / s["count"] if s["count"] else 0,
            }
        result["total_tasks"] = len(self._task_store)
        result["agents_registered"] = len(self._agents)
        return result

    def save_log(self, path: str) -> None:
        records = []
        for task in self._task_store.values():
            payload_bytes = sum(
                p.size_bytes() for m in task.messages for p in m.parts
            )
            artifact_bytes = sum(a.size_bytes for a in task.artifacts)
            records.append({
                "task_id": task.task_id,
                "task_type": task.task_type,
                "sender": task.sender_id,
                "receiver": task.receiver_id,
                "state": task.state.value,
                "created_at": task.created_at,
                "completed_at": task.completed_at,
                "duration_s": (
                    round(task.completed_at - task.created_at, 4)
                    if task.completed_at else None
                ),
                "n_messages": len(task.messages),
                "n_artifacts": len(task.artifacts),
                "payload_bytes": payload_bytes,
                "artifact_bytes": artifact_bytes,
            })

        out = {
            "summary": self.summary(),
            "tasks": records,
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
