
import argparse
import os
from tool_manger import Tool,ToolParameter,ToolRegistry
from typing import List
from harness import Harness
from llm_client import MockLLMClient,OpenAILLMClient
import asyncio
from user_input import UserInterface

def get_weather(city: str) -> str:
    mock_db = {
        "北京": "晴天，25°C，空气质量优",
        "上海": "多云，28°C，湿度65%",
        "广州": "小雨，30°C，湿度80%",
        "深圳": "雷阵雨，29°C，湿度85%",
        "杭州": "阴天，26°C，微风",
    }
    return mock_db.get(city, f"暂无 {city} 的天气数据")


def calculate(expression: str) -> str:
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式包含非法字符"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def read_file(path: str) -> str:
    safe_dir = "/tmp/sandbox"
    abs_path = os.path.abspath(os.path.join(safe_dir, os.path.basename(path)))
    if not abs_path.startswith(safe_dir):
        return "错误：访问被拒绝"
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 {path} 不存在"
    except Exception as e:
        return f"读取错误: {e}"


def query_database(query: str) -> str:
    query_lower = query.lower().strip()
    dangerous = ["drop", "delete", "update", "insert", "alter", "create"]
    if any(d in query_lower for d in dangerous):
        return "错误：检测到危险操作，仅允许 SELECT"
    if not query_lower.startswith("select"):
        return "错误：仅支持 SELECT 查询"
    mock_data = {
        "users": "| id | name | email |\n| 1 | 张三 | zhangsan@example.com |\n| 2 | 李四 | lisi@example.com |",
        "orders": "| order_id | user_id | amount | status |\n| 1001 | 1 | 299.00 | paid |\n| 1002 | 2 | 150.50 | pending |"
    }
    for table, data in mock_data.items():
        if table in query_lower:
            return f"查询结果(表: {table}):\n{data}"
    return "查询成功：返回 0 条记录"


def send_email(to: str, subject: str, body: str) -> str:
    return f"模拟发送邮件成功：\n收件人: {to}\n主题: {subject}\n内容: {body[:50]}..."


def create_sample_tools() -> List[Tool]:
    return [
        Tool(
            name="get_weather",
            description="查询指定城市的当前天气状况",
            parameters=[ToolParameter("city", "string", "城市名称，如北京、上海、广州")],
            func=get_weather,
            tags=["weather"]
        ),
        Tool(
            name="calculate",
            description="计算数学表达式",
            parameters=[ToolParameter("expression", "string", "数学表达式，如 15*23+8")],
            func=calculate,
            tags=["calculate"]
        ),
        Tool(
            name="read_file",
            description="读取文件内容（沙箱限制）",
            parameters=[ToolParameter("path", "string", "文件路径，相对于 /tmp/sandbox")],
            func=read_file,
            tags=["file"]
        ),
        Tool(
            name="query_database",
            description="执行数据库查询（仅支持 SELECT）",
            parameters=[ToolParameter("query", "string", "SQL 查询语句")],
            func=query_database,
            tags = ["database"]
        ),
        Tool(
            name="send_email",
            description="发送电子邮件（危险操作）",
            parameters=[
                ToolParameter("to", "string", "收件人邮箱"),
                ToolParameter("subject", "string", "邮件主题"),
                ToolParameter("body", "string", "邮件正文")
            ],
            func=send_email,
            tags=["email"],
            dangerous=True,
        )
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