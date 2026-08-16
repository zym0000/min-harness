import asyncio
import os
import subprocess
import shutil
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import Completer, Completion

from harness import Harness
from event.event import EventType
from interaction.display import render, R, B, D, WHT, BLU, GRN, YEL, CYN, RED

BANNER = f"""
{CYN}{B}
███╗   ███╗██╗███╗   ██╗      ██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗
████╗ ████║██║████╗  ██║      ██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝
██╔████╔██║██║██╔██╗ ██║█████╗███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗
██║╚██╔╝██║██║██║╚██╗██║╚════╝██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║
██║ ╚═╝ ██║██║██║ ╚████║      ██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║
╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝      ╈█║  ██║██║  ██║██║  ██║██║  ╚══██║╚══════╝╚══════╝╚══════╝
{R}"""

HELP = f"""
{YEL}Commands:{R}
  /help       this help
  /tools      list tools
  /status     metrics & state
  /cancel     cancel running task (also: Ctrl+C during a running task)
  /clear      clear screen + reset current task memory
  /quit       exit

{YEL}Usage:{R}
  Just type what you want the agent to do.
  Dangerous ops (write_file, shell_exec, apply_patch) need approval.
  Multi-turn: keep typing to continue the same task.

{YEL}Multi-line input:{R}
  - Press Enter to submit.
  - Alt+Enter (or Esc, then Enter) to insert a newline.
  - Pasting multi-line content from clipboard is handled as a single block.
  - Ctrl+C cancels current input (returns to prompt).
  - Ctrl+D on empty line exits the CLI.
"""


async def prompt_approval(tool_name: str, arguments: dict) -> tuple[bool, str]:
    """占位审批函数，请替换为实际交互逻辑。"""
    print(f"  {YEL}[APPROVAL] {BLU}{tool_name}{YEL} {arguments}{R}")
    ans = input(f"  {WHT}Approve? (y/n): {R}").strip().lower()
    if ans == 'y':
        return True, ""
    else:
        return False, "user denied"


class CommandCompleter(Completer):
    COMMANDS = [
        '/help', '/tools', '/status', '/cancel',
        '/clear', '/quit', '/exit', '/q', '/metrics'
    ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith('/'):
            for cmd in self.COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))


class InteractiveCLI:
    def __init__(self, harness: Harness, workspace: str):
        self.harness = harness
        self.workspace = workspace
        self.task_id: Optional[str] = None
        self.alive = True
        self._loop = asyncio.get_event_loop()
        self._width = max(40, shutil.get_terminal_size((80, 24)).columns)
        self._session = self._create_prompt_session()

    def _create_prompt_session(self) -> PromptSession:
        """创建多行输入会话：Enter 提交，Alt+Enter 插入换行"""
        kb = KeyBindings()

        # 核心：Enter 提交
        @kb.add('enter')
        def _(event):
            event.current_buffer.validate_and_handle()

        # Alt + Enter 或 Esc -> Enter：插入真正的换行
        @kb.add('escape', 'enter')
        def _(event):
            event.current_buffer.insert_text('\n')

        # Ctrl+C 抛出中断
        @kb.add('c-c')
        def _(event):
            raise KeyboardInterrupt()

        style = Style.from_dict({
            'prompt':       'ansigreen bold',
            'continuation': 'ansibrightblack',
        })

        session = PromptSession(
            multiline=True,
            key_bindings=kb,
            style=style,
            completer=CommandCompleter(),
        )
        return session

    async def _input(self) -> Optional[str]:
        """异步读取多行输入。返回去除首尾空白后的文本；Ctrl+C / Ctrl+D 时返回 None。"""
        try:
            text = await self._loop.run_in_executor(
                None,
                lambda: self._session.prompt(
                    message=[('class:prompt', '❯ '), ('class:continuation', '')],
                    prompt_continuation=lambda width, line_number, is_soft_wrap: [
                        ('class:continuation', '… ')
                    ],
                ),
            )
            return text.strip() if text else None
        except KeyboardInterrupt:
            print(f"\n  {YEL}(Ctrl+C — type /quit to exit){R}")
            return None
        except EOFError:
            return None

    async def run(self):
        """启动 CLI 主循环。"""
        print(BANNER)
        print(f"  {D}workspace: {self.workspace}{R}")
        print(f"  {D}type /help for commands{R}")
        print(f"  {D}Alt+Enter for newline, paste blocks as a whole.{R}\n")

        while self.alive:
            try:
                text = await self._input()
            except KeyboardInterrupt:
                print(f"\n  {YEL}(ctrl+c — /quit to exit){R}")
                continue
            except EOFError:
                break

            if text is None:
                continue

            if text.startswith("/"):
                await self._cmd(text)
            else:
                await self._chat(text)

        print(f"\n  {BLU} bye{R}")

    async def _chat(self, text: str):
        """处理用户输入的普通对话"""
        if self.task_id:
            ok, gen = await self.harness.continue_task(self.task_id, text)
            if not ok:
                self.task_id, gen = await self.harness.submit_task(text)
        else:
            self.task_id, gen = await self.harness.submit_task(text)

        print(f"  {D}task: {self.task_id[:12]}…{R}")

        try:
            async for ev in gen:
                if ev.event_type == EventType.NEED_APPROVAL:
                    render(ev)
                    approved, fb = await prompt_approval(
                        ev.tool_name or "",
                        (ev.data or {}).get("arguments", {}),
                    )
                    if approved:
                        await self.harness.grant_approval(self.task_id)
                    else:
                        await self.harness.reject_approval(self.task_id)
                    continue
                render(ev)
        except (KeyboardInterrupt, asyncio.CancelledError):
            # 任务被用户中断或被系统取消
            if self.task_id:
                try:
                    await self.harness.cancel_task(self.task_id)
                except Exception:
                    pass
            print(f"  {YEL}task cancelled by user{R}")
        except Exception as e:
            print(f"  {RED}stream error: {e}{R}")
            self.task_id = None

    async def _cmd(self, raw: str):
        """解析并执行 CLI 内置命令。"""
        c = raw.lower().strip()

        if c in ("/quit", "/exit", "/q"):
            self.alive = False
        elif c == "/help":
            print(HELP)
        elif c == "/tools":
            tools = self.harness.registry.list_tools()
            print(f"\n  {BLU}{B}Tools ({len(tools)}):{R}")
            for t in tools:
                ap = " " if getattr(t, "requires_approval", False) else ""
                print(f"    {WHT}• {BLU}{t.name}{R}")
            print()
        elif c == "/status":
            print(f"\n  {BLU}task:{R}      {WHT}{self.task_id or 'none'}{R}")
            print(f"  {BLU}workspace:{R} {WHT}{self.workspace}{R}")
            m = self.harness.get_metrics_summary()
            if m:
                print(f"  {D}metrics: {m}{R}")
            print()
        elif c == "/cancel":
            if self.task_id:
                await self.harness.cancel_task(self.task_id)
                print(f"  {YEL}cancelled{R}")
                self.task_id = None
            else:
                print(f"  {D}no active task{R}")
        elif c == "/clear":
            subprocess.run("clear" if os.name != "nt" else "cls",shell=True)
            tid = self.task_id
            if not tid:
                print(f"  {D}no active task — screen cleared{R}")
            else:
                cleared = await self.harness.clear_task(tid)
                self.task_id = None
                if cleared:
                    print(f"  {YEL}task cleared: {tid[:12]}…{R}")
                else:
                    print(f"  {D}task already gone — screen cleared{R}")
        elif c == "/metrics":
            self.harness.print_metrics()
        else:
            print(f"  {YEL}unknown: {c}  (/help){R}")