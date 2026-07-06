
import argparse
import os
from tool_manger import Tool,ToolParameter,ToolRegistry
from typing import List,Optional
from harness import Harness
from llm_client import MockLLMClient
import re
import asyncio
from user_input import UserInterface,_to_thread

async def get_weather(city: str) -> str:
    await asyncio.sleep(0.05)
    mock_db = {"北京": "晴天，25°C", "上海": "多云，28°C", "广州": "小雨，30°C"}
    return mock_db.get(city, f"暂无 {city} 的天气数据")


async def calculate(expression: str) -> str:
    await asyncio.sleep(0.02)
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式包含非法字符"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


async def read_file(path: str) -> str:
    await asyncio.sleep(0.03)
    safe_dir = "/tmp/sandbox"
    abs_path = os.path.abspath(os.path.join(safe_dir, os.path.basename(path)))
    # if not abs_path.startswith(safe_dir):
    #     return "错误：访问被拒绝"
    # if aiofiles is not None:
    #     async with aiofiles.open(abs_path, "r", encoding="utf-8") as f:
    #         return await f.read()
    # else:
    return await _to_thread(_sync_read_file, abs_path)

def _sync_read_file(abs_path: str) -> str:
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 {os.path.basename(abs_path)} 不存在"
    except Exception as e:
        return f"读取错误: {e}"

async def query_database(query: str) -> str:
    await asyncio.sleep(0.05)
    query_lower = query.lower().strip()
    dangerous = ["drop", "delete", "update", "insert", "alter", "create"]
    if any(d in query_lower for d in dangerous):
        return "错误：检测到危险操作，仅允许 SELECT"
    if not query_lower.startswith("select"):
        return "错误：仅支持 SELECT 查询"
    return "查询成功：返回 0 条记录"

async def send_email(to: str, subject: str, body: str) -> str:
    await asyncio.sleep(0.05)
    return f"模拟发送邮件成功：\n收件人: {to}\n主题: {subject}"

async def str_replace_editor(command: str, path: str, old_str: str = "", new_str: str = "", file_text: str = "", view_range: Optional[List[int]] = None) -> str:
    await asyncio.sleep(0.03)
    safe_dir = "/tmp/sandbox"
    abs_path = os.path.abspath(os.path.join(safe_dir, os.path.basename(path)))
    if not abs_path.startswith(safe_dir):
        return "错误：访问被拒绝（超出沙箱范围）"
    if command == "view":
        if not os.path.exists(abs_path):
            return f"错误：文件 {path} 不存在"
        if os.path.isdir(abs_path):
            files = os.listdir(abs_path)
            return f"目录 {path}:\n" + "\n".join(f"  {f}" for f in files[:50])
        content = await _to_thread(_sync_read_file, abs_path)
        if view_range and len(view_range) == 2:
            lines = content.split("\n")
            start, end = view_range[0] - 1, view_range[1]
            content = "\n".join(lines[start:end])
        return content
    elif command == "create":
        if os.path.exists(abs_path):
            return f"错误：文件 {path} 已存在"
        await _to_thread(_sync_write_file, abs_path, file_text)
        return f"文件 {path} 已创建"
    elif command == "str_replace":
        if not os.path.exists(abs_path):
            return f"错误：文件 {path} 不存在"
        content = await _to_thread(_sync_read_file, abs_path)
        if old_str not in content:
            return "错误：未找到匹配字符串，请确保 old_str 唯一存在"
        if content.count(old_str) > 1:
            return "错误：匹配字符串不唯一，请提供更具体的上下文"
        new_content = content.replace(old_str, new_str, 1)
        await _to_thread(_sync_write_file, abs_path, new_content)
        return f"文件 {path} 已更新"
    elif command == "insert":
        if not os.path.exists(abs_path):
            return f"错误：文件 {path} 不存在"
        content = await _to_thread(_sync_read_file, abs_path)
        lines = content.split("\n")
        insert_line = int(view_range[0]) if view_range else len(lines)
        lines.insert(insert_line, new_str)
        await _to_thread(_sync_write_file, abs_path, "\n".join(lines))
        return f"已在 {path} 第 {insert_line + 1} 行插入内容"
    return f"错误：未知命令 {command}"

def _sync_write_file(abs_path: str, content: str) -> None:
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)

async def bash(command: str, timeout: float = 30.0) -> str:
    await asyncio.sleep(0.05)
    dangerous_patterns = [
        r"rm\s+-rf", r"rm\s+/", r":(){:|:&};:", r"> /dev/sda",
        r"mkfs", r"dd if=", r"chmod\s+-R\s+777\s+/",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return f"错误：检测到危险命令模式，已阻止执行: {command}"
    return f"$ {command}\n模拟输出：命令执行成功（exit code 0）\n（实际生产环境应使用 subprocess 执行）"

async def grep_search(query: str, path: str = ".") -> str:
    await asyncio.sleep(0.05)
    safe_dir = "/tmp/sandbox"
    abs_path = os.path.abspath(os.path.join(safe_dir, os.path.basename(path)))
    if not abs_path.startswith(safe_dir):
        return "错误：访问被拒绝"
    if not os.path.exists(abs_path):
        return f"错误：路径 {path} 不存在"
    results = []
    try:
        for root, dirs, files in os.walk(abs_path):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.java', '.go', '.rs', '.c', '.cpp', '.h')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for i, line in enumerate(f, 1):
                                if re.search(query, line, re.IGNORECASE):
                                    rel_path = os.path.relpath(file_path, safe_dir)
                                    results.append(f"{rel_path}:{i}: {line.strip()}")
                                    if len(results) >= 20:
                                        break
                            if len(results) >= 20:
                                break
                    except Exception:
                        continue
            if len(results) >= 20:
                break
    except Exception as e:
        return f"搜索错误: {e}"
    if not results:
        return f"未找到匹配 '{query}' 的结果"
    return f"找到 {len(results)} 个匹配：\n" + "\n".join(results[:20])

def create_sample_tools() -> List[Tool]:
    return [
        Tool("get_weather", "查询指定城市的当前天气状况", [ToolParameter("city", "string", "城市名称")], get_weather, tags=["get_weather"]),
        Tool("calculate", "计算数学表达式", [ToolParameter("expression", "string", "数学表达式")], calculate, tags=["calculate"]),
        Tool("read_file", "读取文件内容（沙箱限制）", [ToolParameter("path", "string", "文件路径")], read_file, tags=["read_file"]),
        Tool("query_database", "执行数据库查询（仅支持 SELECT）", [ToolParameter("query", "string", "SQL 查询语句")], query_database, tags=["query_database"]),
        Tool("send_email", "发送电子邮件（危险操作）", [
            ToolParameter("to", "string", "收件人邮箱"),
            ToolParameter("subject", "string", "邮件主题"),
            ToolParameter("body", "string", "邮件正文"),
        ], send_email, dangerous=True, tags=["send_email"]),
        Tool("str_replace_editor", "文件编辑器", [
            ToolParameter("command", "string", "操作: view/create/str_replace/insert"),
            ToolParameter("path", "string", "文件路径"),
            ToolParameter("old_str", "string", "被替换字符串", required=False),
            ToolParameter("new_str", "string", "新字符串", required=False),
            ToolParameter("file_text", "string", "文件内容（create 时必填）", required=False),
            ToolParameter("view_range", "array", "查看范围（view/insert 时可选）", required=False),
        ], str_replace_editor, dangerous=True, tags=["str_replace_editor"]),
        Tool("bash", "执行 shell 命令（危险操作，需确认）", [
            ToolParameter("command", "string", "shell 命令"),
            ToolParameter("timeout", "number", "超时时间（秒）", required=False),
        ], bash, dangerous=True, tags=["bash"]),
        Tool("grep_search", "搜索代码", [
            ToolParameter("query", "string", "搜索正则"),
            ToolParameter("path", "string", "搜索路径", required=False),
        ], grep_search, tags=["grep_search"]),
    ]

async def demo_async_mode():
    print("\n" + "Async 测试" + "\n")

    registry = ToolRegistry()
    for tool in create_sample_tools():
        registry.register(tool)

    llm = MockLLMClient(registry)
    harness = Harness(registry, llm, max_concurrent=10)
    ui = UserInterface(harness)

    await ui.start_interactive()

    harness.print_metrics()

def main():
    parser = argparse.ArgumentParser(description="Minimal Harness")
    parser.add_argument("--real", action="store_true", help="使用真实 LLM(暂未实现)")
    args = parser.parse_args()

    asyncio.run(demo_async_mode())

if __name__ == "__main__":
    main()