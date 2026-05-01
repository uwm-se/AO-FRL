"""A2A (Agent-to-Agent) protocol data structures.

Follows the Google A2A specification for inter-agent communication,
adapted for in-process federated learning agent coordination.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskState(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCard:
    agent_id: str
    name: str
    description: str
    skills: list[str] = field(default_factory=list)


@dataclass
class Part:
    type: str       # "text" | "data" | "json"
    content: Any    # str / torch.Tensor / dict

    def size_bytes(self) -> int:
        import torch
        if isinstance(self.content, torch.Tensor):
            return self.content.nelement() * self.content.element_size()
        elif isinstance(self.content, dict):
            import json
            return len(json.dumps(self.content, default=str).encode())
        elif isinstance(self.content, str):
            return len(self.content.encode())
        return 0


@dataclass
class Message:
    role: str           # "user" | "agent"
    sender_id: str
    parts: list[Part] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Artifact:
    artifact_id: str
    name: str
    data: Any           # tensor, state_dict, etc.
    mime_type: str = "application/octet-stream"
    size_bytes: int = 0


@dataclass
class Task:
    task_id: str
    task_type: str      # "extract_embeddings" | "local_train" | "orchestrate" | ...
    sender_id: str
    receiver_id: str
    state: TaskState = TaskState.PENDING
    messages: list[Message] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
