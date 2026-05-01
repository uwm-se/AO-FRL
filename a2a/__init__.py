"""A2A (Agent-to-Agent) protocol layer for federated learning agent communication."""

from .protocol import AgentCard, Artifact, Message, Part, Task, TaskState
from .bus import A2ABus

__all__ = [
    "AgentCard", "Artifact", "Message", "Part", "Task", "TaskState",
    "A2ABus",
]
