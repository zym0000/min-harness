import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from skill.skill import ProgressiveSkillManager
from tool_manger import Tool, ToolParameter, ToolRegistry

_LOG = logging.getLogger("harness.unified")

class UnifiedToolRegistry:
    def __init__(self,
                 skill_manager: Optional[ProgressiveSkillManager] = None,
                 base_registry: Optional[ToolRegistry] = None):
        self.registry: ToolRegistry = base_registry or ToolRegistry()
        self.skill_manager = skill_manager
        self._gated: Set[str] = set()               # 受技能门控的工具名
        # 任务运行态：task_id -> {"skill_name", "current_stage"}
        self._task_states: Dict[str, Dict[str, str]] = {}
        self._state_lock = asyncio.Lock()
        if skill_manager is not None:
            self._register_activate_tool()

    def register(self, tool: Tool, gated: bool = False):
        """注册本地工具；gated=True 表示受技能门控（未激活技能时隐藏）。"""
        self.registry.register(tool)
        if gated:
            self._gated.add(tool.name)

    def register_mcp_tools(self, mcp_tools: List[Any]):
        """把 McpWrappedTool 列表适配为 Tool 并注册（默认门控 + tag mcp）。"""
        for wt in mcp_tools:
            tags = list(getattr(wt, "tags", None) or ["mcp"])
            tool = Tool(
                name=wt.name,
                description=wt.description,
                parameters=wt.parameters,
                func=wt.execute,                 # McpWrappedTool.execute(**kwargs)
                tags=tags,
                executor_type="async",
            )
            self.registry.register(tool)
            self._gated.add(tool.name)
        _LOG.info("MCP 工具注册完成：%d 个（门控）", len(mcp_tools))

    def _register_activate_tool(self):
        async def activate_skill(skill_name: str, task_id: str) -> str:
            error = await self.initialize_task_context(task_id, skill_name)
            if error:
                return f"[Error] {error}"
            return self.skill_manager.activate_skill(skill_name)

        self.registry.register(Tool(
            name="activate_skill",
            description=("激活一个技能（使用前必须激活）。激活后按技能引导执行，"
                         "并通过调用与目标阶段同名的工具推进阶段。"),
            parameters=[
                ToolParameter("skill_name", "string", "技能名", required=True),
            ],
            func=activate_skill,
            tags=["general", "skill"],
            executor_type="async",
        ))
        # 标记需要引擎注入 task_id（LLM 只传 skill_name）
        self.registry.get("activate_skill").needs_task_id = True

    def get(self, name: str) -> Tool:
        return self.registry.get(name)

    def list_tools(self) -> List[Tool]:
        return self.registry.list_tools()

    def validate_call(self, call) -> Optional[str]:
        return self.registry.validate_call(call)

    def describe_tools(self, tools: Optional[List[Tool]] = None) -> str:
        return self.registry.describe_tools(tools)

    def to_openai_schema(self, tools: Optional[List[Tool]] = None):
        return self.registry.to_openai_schema(tools)

    def filter_by_tags(self, tags: List[str]) -> List[Tool]:
        return self.registry.filter_by_tags(tags)

    def get_discovery_prompt_patch(self) -> str:
        if self.skill_manager is None:
            return ""
        return self.skill_manager.get_discovery_system_prompt_patch()

    async def get_disclosed_tools_for_task(self, task_id: str) -> List[Tool]:
        """渐进披露：未激活只给非门控工具；激活后叠加当前 stage 允许的门控工具。"""
        all_tools = self.registry.list_tools()
        always = [t for t in all_tools if t.name not in self._gated]

        if self.skill_manager is None:
            return all_tools

        async with self._state_lock:
            ctx = self._task_states.get(task_id)
            if not ctx:
                return always
            skill_name, stage = ctx["skill_name"], ctx["current_stage"]

        allowed = self.skill_manager.get_allowed_tools_for_stage(skill_name, stage)
        if "*" in allowed:
            return all_tools
        gated_allowed = [self.registry.get(n) for n in allowed
                         if n in self._gated and n in
                         {t.name for t in all_tools}]
        
        seen = {t.name for t in always}
        disclosed = list(always)
        for t in gated_allowed:
            if t.name not in seen:
                disclosed.append(t)
        return disclosed
    
    async def initialize_task_context(self, task_id: str, skill_name: str) -> Optional[str]:
        """激活技能：初始化任务运行态。返回 None 成功，否则错误消息。"""
        initial = self.skill_manager.get_initial_stage(skill_name)
        if initial is None:
            return f"skill '{skill_name}' not found"
        async with self._state_lock:
            self._task_states[task_id] = {
                "skill_name": skill_name, "current_stage": initial}
        return None

    async def transit_task_skill_state(self, task_id: str, target_stage: str) -> Optional[str]:
        """隐式转移：agent_loop 在工具执行成功后以工具名为 target_stage 调用。"""
        if not target_stage:
            return "missing target_stage parameter"

        async with self._state_lock:
            ctx = self._task_states.get(task_id)
            if not ctx:
                return None
            skill_name, current = ctx["skill_name"], ctx["current_stage"]

        skill = self.skill_manager.skills_pool.get(skill_name)
        if skill is None or target_stage not in skill.stages:
            return None

        valid = self.skill_manager.get_valid_transitions(skill_name, current)
        if target_stage not in valid:
            return (f"不能从 {current} 转移到 {target_stage}，"
                    f"允许的目标：{valid}")

        async with self._state_lock:
            if task_id in self._task_states:
                self._task_states[task_id]["current_stage"] = target_stage
                _LOG.info("任务 %s 技能 %s: %s -> %s",
                          task_id, skill_name, current, target_stage)
        return None

    async def on_task_end(self, task_id: str):
        """Harness 生命周期回调：任务结束清理运行态。"""
        async with self._state_lock:
            self._task_states.pop(task_id, None)

    # 测试/调试辅助
    async def get_task_stage(self, task_id: str) -> Optional[str]:
        async with self._state_lock:
            ctx = self._task_states.get(task_id)
            return ctx["current_stage"] if ctx else None