"""Bootstrap 组装"""
import os
from pathlib import Path
from typing import Optional

from tools.mcp_wrapped_tool import create_pool_and_discover
from skill.registry import UnifiedToolRegistry
from skill.skill import ProgressiveSkillManager
from harness import Harness, HarnessConfig
from config.loader import resolve_llm, resolve_paths
from llm_client import LLMClient
from subagent import make_sub_agent_tool

async def bootstrap(
    workspace: str = "./workspace",
    skills_dir: str = "./skills",
    mcp_server_script: Optional[str] = None,
    model: Optional[str] = None,
    store_path: Optional[str] = None,
    max_steps: int = 200,
) -> Harness:

    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    os.environ["AGENT_WORKSPACE"] = str(ws)

    if mcp_server_script is None:
        mcp_server_script = str(
            Path(__file__).parent.parent / "tools" / "coding_tools_server.py"
        )

    resolved = resolve_paths(
        default_skills_dir=skills_dir,
        default_mcp_server_script=mcp_server_script
            or str(Path(__file__).parent.parent / "tools" / "coding_tools_server.py"),
    )
    print(f"  [bootstrap] resolved paths:")
    print(f"    skills_dir:        {resolved.skills_dir}")
    print(f"    mcp_server_script: {resolved.mcp_server_script}")

    resolved_llm = resolve_llm(
        default_api_key=None,
        default_base_url="https://api.minimaxi.com/v1",
        default_model=model or "MiniMax-M3",
    )
    print(f"  [bootstrap] resolved llm:")
    print(f"    base_url: {resolved_llm.base_url}")
    print(f"    model:    {resolved_llm.model}")
    print(f"    api_key:  {'<set>' if resolved_llm.api_key else '<unset>'}")

    _, wrapped_tools = await create_pool_and_discover(
        server_script=str(resolved.mcp_server_script),
        server_name="coding-tools",
        mode="InMemoryStack",
        mcp_variable_name="mcp",
        # include={"read_file", "write_file", "apply_patch", ...},  # 可选过滤
        # exclude={"file_info"},                                     # 可选排除
    )
    print(f"  [bootstrap] {len(wrapped_tools)} tools: "
          f"{[t.name for t in wrapped_tools]}")

    skill_mgr = ProgressiveSkillManager(skills_dir=str(resolved.skills_dir))
    skill_mgr.load_all_skills()
    print(f"  [bootstrap] {len(skill_mgr.skills_pool)} skills: "
          f"{list(skill_mgr.skills_pool.keys())}")

    tool_registry = UnifiedToolRegistry(skill_manager=skill_mgr)
    tool_registry.register_mcp_tools(wrapped_tools)

    llm = LLMClient(
        api_key=resolved_llm.api_key,
        base_url=resolved_llm.base_url,
        model=resolved_llm.model,
        timeout=120.0,
    )

    cfg = HarnessConfig(
        max_concurrent=5,
        max_steps=max_steps,
        llm_timeout=120.0,
        approval_timeout=300.0,
        max_consecutive_parse_errors=3,
        enable_watchdog=True,
        context_max_tokens=100000,
        context_reserve_tokens=10000,
        store_path=store_path,
    )

    harness = Harness(registry=tool_registry, llm=llm, config=cfg)

    # 注册 sub_agent 内置工具（非门控、tag=general）。闭包捕获 harness,
    # 使 Tool.func 能访问 _loop_tools / _subagent_depth / approval_gates 等。
    tool_registry.register(make_sub_agent_tool(harness))

    restored = await harness.restore_tasks()
    if restored:
        print(f"  [bootstrap] restored {restored} tasks")

    return harness