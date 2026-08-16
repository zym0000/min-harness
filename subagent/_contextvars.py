"""Sub-agent 内部共享的 ContextVar 定义(深度 + 嵌套 parent 上下文)。

只有 _current_event_sink 例外,它定义在 agent_loop.py 中——因为 sink 的 owner 是
AgentLoop.run(),subagent/runner.py 只是 consumer。把 sink 放在 agent_loop.py 里,
依赖方向是 runner → agent_loop,符合现有 import 关系。
"""
from __future__ import annotations

import contextvars
from typing import Any, Optional

_current_depth: contextvars.ContextVar = contextvars.ContextVar(
    "subagent_current_depth",
    default=0,
)

_current_context: contextvars.ContextVar = contextvars.ContextVar(
    "subagent_current_context",
    default=None,
)

__all__ = [
    "_current_depth",
    "_current_context",
]
