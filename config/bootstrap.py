"""Bootstrap 组装"""
import os
from pathlib import Path
from typing import Optional

from tools.mcp_wrapped_tool import create_pool_and_discover
from skill.registry import UnifiedToolRegistry
from skill.skill import ProgressiveSkillManager
from harness import Harness, HarnessConfig
from llm_client import LLMClient

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

    _, wrapped_tools = await create_pool_and_discover(
        server_script=mcp_server_script,
        server_name="coding-tools",
        mode="InMemoryStack",
        mcp_variable_name="mcp",
        # include={"read_file", "write_file", "apply_patch", ...},  # 可选过滤
        # exclude={"file_info"},                                     # 可选排除
    )
    print(f"  [bootstrap] {len(wrapped_tools)} tools: "
          f"{[t.name for t in wrapped_tools]}")

    skill_mgr = ProgressiveSkillManager(skills_dir=skills_dir)
    skill_mgr.load_all_skills()
    print(f"  [bootstrap] {len(skill_mgr.skills_pool)} skills: "
          f"{list(skill_mgr.skills_pool.keys())}")

    tool_registry = UnifiedToolRegistry(skill_manager=skill_mgr)
    tool_registry.register_mcp_tools(wrapped_tools)

    llm = LLMClient(
        api_key=os.environ.get("OPENAI_API_KEY", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        model=model or os.environ.get("AGENT_MODEL", "deepseek-v4-flash"),
        timeout=120.0,
    )

    cfg = HarnessConfig(
        max_concurrent=5,
        max_steps=max_steps,
        llm_timeout=120.0,
        approval_timeout=300.0,
        max_consecutive_parse_errors=3,
        enable_watchdog=True,
        context_max_tokens=12000,
        context_reserve_tokens=3000,
        store_path=store_path,
    )

    harness = Harness(registry=tool_registry, llm=llm, config=cfg)

    restored = await harness.restore_tasks()
    if restored:
        print(f"  [bootstrap] restored {restored} tasks")

    return harness