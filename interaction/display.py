"""终端事件渲染"""
from event.event import EventType, LoopEvent

# ANSI
R = "\033[0m"
B = "\033[1m"
D = "\033[2m"
RED = "\033[91m"
GRN = "\033[92m"
YEL = "\033[93m"
BLU = "\033[94m"
MAG = "\033[95m"
CYN = "\033[96m"

def render(event: LoopEvent):
    t = event.event_type

    if t == EventType.THINKING_STARTED:
        print(f"  {D}{CYN}⟳ thinking…{R}")

    elif t == EventType.THINKING_COMPLETED:
        text = event.content or ""
        # 提取 Thought
        for line in text.splitlines():
            if line.startswith("Thought:"):
                print(f"  {CYN} {line[8:].strip()}{R}")
                break

    elif t == EventType.TOOL_EXECUTION_STARTED:
        name = event.tool_name or "?"
        args = ""
        if event.data:
            args = str(event.data.get("arguments", ""))[:150]
        print(f"  {BLU} {name}({args}){R}")

    elif t == EventType.TOOL_EXECUTION_COMPLETED:
        name = event.tool_name or "?"
        ms = (event.data or {}).get("latency_ms", 0)
        print(f"  {D}{GRN}✓ {name} ({ms:.0f}ms){R}")

    elif t == EventType.TOOL_EXECUTION_FAILED:
        name = event.tool_name or "?"
        err = (event.data or {}).get("error", "")[:200]
        print(f"  {RED}✗ {name}: {err}{R}")

    elif t == EventType.NEED_APPROVAL:
        name = event.tool_name or "?"
        args = (event.data or {}).get("arguments", {})
        print(f"\n  {YEL}{'━' * 56}{R}")
        print(f"  {YEL}{B}  APPROVAL: {name}{R}")
        print(f"  {YEL}{str(args)[:300]}{R}")
        print(f"  {YEL}{'━' * 56}{R}")

    elif t == EventType.APPROVAL_GRANTED:
        print(f"  {GRN}✓ approved{R}")

    elif t == EventType.APPROVAL_DENIED:
        print(f"  {RED}✗ rejected{R}")

    elif t == EventType.FINAL_ANSWER:
        print(f"  {GRN}{B}  Answer:{R}")
        for ln in (event.content or "").splitlines():
            print(f"  {GRN}  {ln}{R}")