"""终端事件渲染"""
from event.event import EventType, LoopEvent

# ANSI
R = "\033[0m"
B = "\033[1m"
D = "\033[2m"
WHT = "\033[97m"
RED = "\033[91m"
GRN = "\033[92m"
YEL = "\033[93m"
BLU = "\033[94m"
MAG = "\033[95m"
CYN = "\033[96m"

def render(event: LoopEvent):
    t = event.event_type

    if t == EventType.THINKING_STARTED:
        print(f"  {D}{BLU}⟳ thinking…{R}")

    elif t == EventType.THINKING_COMPLETED:
        text = event.content or ""
        # 提取 Thought
        for line in text.splitlines():
            if line.startswith("Thought:"):
                print(f"  {WHT} {line[8:].strip()}{R}")
                break

    elif t == EventType.TOOL_EXECUTION_STARTED:
        name = event.tool_name or "?"
        args = ""
        if event.data:
            args = str(event.data.get("arguments", ""))[:150]
        print(f"  {BLU}{B} {name}{R}{WHT}({args}){R}")

    elif t == EventType.TOOL_EXECUTION_COMPLETED:
        name = event.tool_name or "?"
        ms = (event.data or {}).get("latency_ms", 0)
        print(f"  {WHT}✓ {BLU}{name}{WHT} ({D}{ms:.0f}ms{R}){R}")

    elif t == EventType.TOOL_EXECUTION_FAILED:
        name = event.tool_name or "?"
        # 优先取 data.error(标准 TOOL_EXECUTION_FAILED),fallback 到顶层 event.error
        # (sub-agent runner 的 ERROR 事件把 error 放在顶层字段)
        err = (event.data or {}).get("error")
        if err is None:
            err = event.error
        err = str(err or "")[:200]
        print(f"  {RED}✗ {name}: {err}{R}")

    elif t == EventType.NEED_APPROVAL:
        name = event.tool_name or "?"
        args = (event.data or {}).get("arguments", {})
        print(f"\n  {YEL}{'━' * 56}{R}")
        print(f"  {YEL}{B}  APPROVAL: {name}{R}")
        print(f"  {WHT}{str(args)[:300]}{R}")
        print(f"  {YEL}{'━' * 56}{R}")

    elif t == EventType.APPROVAL_GRANTED:
        print(f"  {BLU}✓ approved{R}")

    elif t == EventType.APPROVAL_DENIED:
        print(f"  {RED}✗ rejected{R}")

    elif t == EventType.FINAL_ANSWER:
        print(f"  {BLU}{B}  Answer:{R}")
        for ln in (event.content or "").splitlines():
            print(f"  {WHT}  {ln}{R}")