"""
事件系统：Agent 之间的异步通信机制
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
import uuid


class EventType(Enum):
    """事件类型枚举"""
    TASK_DECOMPOSED = "task_decomposed"
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_COMPLETED = "subtask_completed"
    CONTEXT_UPDATED = "context_updated"
    AGENT_QUESTION = "agent_question"
    AGENT_ANSWER = "agent_answer"
    SWARM_STARTED = "swarm_started"
    SWARM_COMPLETED = "swarm_completed"


@dataclass
class Event:
    """
    事件数据类

    Agent 通过发布事件到 SharedContext 来通信，而不是直接调用其他 Agent。
    """
    type: EventType
    source_agent: str
    data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    target_agents: Optional[List[str]] = None

    def is_for_agent(self, agent_id: str) -> bool:
        """判断事件是否针对特定 Agent"""
        if self.target_agents is None:
            return True
        return agent_id in self.target_agents

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "type": self.type.value,
            "source_agent": self.source_agent,
            "timestamp": self.timestamp.isoformat(),
            "target_agents": self.target_agents,
            "data": self.data,
        }
