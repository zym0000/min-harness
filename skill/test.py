import asyncio
import os
import sys
import tempfile

sys.path.insert(0, "/tmp/harness_pkg")

PASS, FAIL = 0, 0

def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


SKILL_YAML = """---
name: dev_flow
description: 开发流程技能
initial_stage: coding
stages:
  coding:
    description: 写代码阶段
    allowed_tools: [write_code, review]
  review:
    description: 评审阶段
    allowed_tools: [approve]
transitions:
  coding: [review]
---
按 coding -> review 的流程执行。
"""

BAD_SKILL = """---
name: bad_flow
description: 转移目标不存在
stages:
  a:
    allowed_tools: []
transitions:
  a: [ghost_stage]
---
"""


def make_skills_dir(extra_files=None):
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "dev_flow"))
    with open(os.path.join(d, "dev_flow", "SKILL.md"), "w") as f:
        f.write(SKILL_YAML)
    for name, content in (extra_files or {}).items():
        os.makedirs(os.path.join(d, name))
        with open(os.path.join(d, name, "SKILL.md"), "w") as f:
            f.write(content)
    return d


# ---------------- skill_manager 解析 ----------------

async def t_skill_parsing():
    from skill import ProgressiveSkillManager
    print("\n== skill_manager 解析 ==")

    d = make_skills_dir({"bad_flow": BAD_SKILL})     # 一个坏技能
    sm = ProgressiveSkillManager(d)
    sm.load_all_skills()                              # 不应被坏技能中断
    check("好技能加载成功", "dev_flow" in sm.skills_pool)
    check("坏技能被隔离跳过", "bad_flow" not in sm.skills_pool)

    skill = sm.skills_pool["dev_flow"]
    check("YAML initial_stage 被采用", skill.initial_stage == "coding")
    check("终态推导正确", skill.terminal_stages == {"review"})
    check("stage 工具集正确",
          sm.get_allowed_tools_for_stage("dev_flow", "coding")
          == {"write_code", "review"})

    # Markdown 表格形态
    d2 = make_skills_dir()
    os.makedirs(os.path.join(d2, "table_skill"))
    with open(os.path.join(d2, "table_skill", "SKILL.md"), "w") as f:
        f.write("| name | table_skill |\n| description | 表格技能 |\n正文引导")
    sm2 = ProgressiveSkillManager(d2)
    sm2.load_all_skills()
    check("表格技能解析", "table_skill" in sm2.skills_pool
          and sm2.skills_pool["table_skill"].description == "表格技能")
    check("软约束包装为单阶段全放行",
          sm2.get_allowed_tools_for_stage("table_skill", "active_mode") == {"*"})

    # 发现 prompt：items() 修复 + activate_skill 契约
    patch = sm.get_discovery_system_prompt_patch()
    check("发现 patch 不崩且含技能", "dev_flow" in patch and "开发流程技能" in patch)
    check("patch 引用已注册的工具名", "activate_skill" in patch)


# ---------------- 统一注册表：披露 ----------------

async def t_disclosure():
    from skill import ProgressiveSkillManager
    from tool_manger import Tool, ToolParameter
    from registry import UnifiedToolRegistry
    print("\n== 渐进披露 ==")

    sm = ProgressiveSkillManager(make_skills_dir())
    sm.load_all_skills()
    reg = UnifiedToolRegistry(sm)

    async def write_code(code: str) -> str:
        return "ok"
    async def review(note: str = "") -> str:
        return "进入评审"
    async def approve(ok: bool = True) -> str:
        return "通过"
    async def echo(text: str) -> str:
        return text

    reg.register(Tool("write_code", "写代码",
                      [ToolParameter("code", "string", "代码", required=True)],
                      write_code, tags=[], executor_type="async"), gated=True)
    reg.register(Tool("review", "提交评审", [], review, tags=[], executor_type="async"),
                 gated=True)
    reg.register(Tool("approve", "批准", [], approve, tags=[], executor_type="async"),
                 gated=True)
    reg.register(Tool("echo", "回显",
                      [ToolParameter("text", "string", "文本", required=True)],
                      echo, tags=["general"], executor_type="async"))

    names = {t.name for t in await reg.get_disclosed_tools_for_task("task1")}
    check("未激活：只给非门控+activate_skill",
          names == {"echo", "activate_skill"}, f"names={names}")

    err = await reg.initialize_task_context("task1", "dev_flow")
    check("激活成功", err is None)
    check("初始阶段为 coding", await reg.get_task_stage("task1") == "coding")

    names2 = {t.name for t in await reg.get_disclosed_tools_for_task("task1")}
    check("激活后披露当前阶段工具", names2
          == {"echo", "activate_skill", "write_code", "review"}, f"names2={names2}")
    check("下阶段工具仍隐藏", "approve" not in names2)

    err = await reg.initialize_task_context("task2", "ghost_skill")
    check("激活不存在技能报错", err is not None)


# ---------------- 统一注册表：转移 ----------------

async def t_transit():
    from skill import ProgressiveSkillManager
    from registry import UnifiedToolRegistry
    print("\n== 隐式转移 ==")

    sm = ProgressiveSkillManager(make_skills_dir())
    sm.load_all_skills()
    reg = UnifiedToolRegistry(sm)

    check("未激活时普通工具调用不受影响",
          await reg.transit_task_skill_state("t1", "whatever") is None)

    await reg.initialize_task_context("t1", "dev_flow")
    check("普通工具名（非阶段名）放行",
          await reg.transit_task_skill_state("t1", "write_code") is None)
    check("阶段名转移成功",
          await reg.transit_task_skill_state("t1", "review") is None)
    check("已进入 review", await reg.get_task_stage("t1") == "review")

    # review 是终态，再调一次 review -> 拒绝且消息正确
    err = await reg.transit_task_skill_state("t1", "review")
    check("非法转移被拒绝", err is not None and "review" in err,
          f"err={err}")

    await reg.on_task_end("t1")
    check("on_task_end 清理状态", await reg.get_task_stage("t1") is None)


# ---------------- 引擎 task_id 注入 ----------------

async def t_task_id_injection():
    from async_execution_engine import AsyncExecutionEngine
    from tool_manger import Tool, ToolParameter
    print("\n== needs_task_id 注入 ==")

    captured = {}

    async def probe(task_id: str, x: str) -> str:
        captured["task_id"] = task_id
        captured["x"] = x
        return "ok"

    tool = Tool("probe", "探针",
                [ToolParameter("x", "string", "参数", required=True)],
                probe, tags=[], executor_type="async")
    tool.needs_task_id = True

    engine = AsyncExecutionEngine()
    result = await engine.execute("task-42", tool, {"x": "1"})
    check("task_id 被注入", captured.get("task_id") == "task-42"
          and captured.get("x") == "1")
    check("执行成功", not result.is_error)


# ---------------- 端到端：Harness + skill ----------------

async def t_end_to_end():
    from event.event import EventType
    from harness import Harness
    from skill import ProgressiveSkillManager
    from tool_manger import Tool, ToolParameter
    from registry import UnifiedToolRegistry
    print("\n== 端到端：激活->执行->转移->收尾 ==")

    sm = ProgressiveSkillManager(make_skills_dir())
    sm.load_all_skills()
    reg = UnifiedToolRegistry(sm)

    calls = []

    async def write_code(code: str) -> str:
        calls.append("write_code")
        return "代码已写入"
    async def review() -> str:
        calls.append("review")
        return "已进入评审阶段"
    async def approve() -> str:
        calls.append("approve")
        return "评审通过"

    reg.register(Tool("write_code", "写代码",
                      [ToolParameter("code", "string", "代码", required=True)],
                      write_code, tags=[], executor_type="async"), gated=True)
    reg.register(Tool("review", "提交评审", [], review, tags=[], executor_type="async"),
                 gated=True)
    reg.register(Tool("approve", "批准", [], approve, tags=[], executor_type="async"),
                 gated=True)

    script = [
        'Thought: 先激活技能\nAction: activate_skill\nAction Input: {"skill_name": "dev_flow"}',
        'Thought: 写代码\nAction: write_code\nAction Input: {"code": "print(1)"}',
        'Thought: 提交评审\nAction: review\nAction Input: {}',
        'Thought: 批准\nAction: approve\nAction Input: {}',
        'Final Answer: 开发流程已走完',
    ]

    class ScriptedLLM:
        def __init__(self):
            self.round = 0
        async def chat(self, messages, tools=None):
            self.round += 1
            return script[min(self.round - 1, len(script) - 1)]

    h = Harness(reg, ScriptedLLM())
    tid, gen = await h.submit_task("帮我完成开发流程")
    completed = False
    async for ev in gen:
        if ev.event_type == EventType.TASK_COMPLETED:
            completed = True
    check("任务完成", completed)
    check("工具按序调用", calls == ["write_code", "review", "approve"],
          f"calls={calls}")
    check("任务结束状态已清理", await reg.get_task_stage(tid) is None)
    st = await h.task_manager.get_state(tid)
    check("激活反馈进了历史", any("activated" in m.get("content", "")
                                    for m in st.messages))


async def main():
    await t_skill_parsing()
    await t_disclosure()
    await t_transit()
    await t_task_id_injection()
    await t_end_to_end()
    print(f"\n==== 结果: {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)

if __name__ == "__main__":
    asyncio.run(main())