

import re
from tool_manger import ToolRegistry
from typing import List,Dict,Optional
import asyncio

class LLMClient:
    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None):
        return NotImplemented


class MockLLMClient(LLMClient):
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def chat(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None
    ) -> str:
        await asyncio.sleep(0.1)
        for msg in reversed(messages):
            content = msg.get("content", "")
            if any(marker in content for marker in ["[Tool Result]", "[Tool Error]", "[Parse Error]", "[Approval Denied]"]):
                return self._generate_summary(content)

        user_input = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_input = msg.get("content", "").lower()
                break
        
        if "天气" in user_input or "weather" in user_input:
            city = self._extract_city(user_input)
            return f'Thought: 用户想查询天气。\nAction: get_weather\nAction Input: {{"city": "{city}"}}'
        elif "计算" in user_input or any(op in user_input for op in ["+", "-", "*", "/"]):
            expr = self._extract_expr(user_input)
            return f'Thought: 用户需要计算。\nAction: calculate\nAction Input: {{"expression": "{expr}"}}'
        elif "文件" in user_input or "read" in user_input:
            return 'Thought: 用户想读取文件。\nAction: read_file\nAction Input: {"path": "example.txt"}'
        elif "数据库" in user_input or "query" in user_input or "搜索" in user_input:
            return 'Thought: 用户需要查询数据库。\nAction: query_database\nAction Input: {"query": "SELECT * FROM users"}'
        elif "邮件" in user_input or "email" in user_input:
            return 'Thought: 用户想发送邮件。\nAction: send_email\nAction Input: {"to": "user@example.com", "subject": "测试", "body": "测试邮件"}'
        elif "编辑" in user_input or "修改" in user_input or "replace" in user_input:
            return 'Thought: 用户想编辑文件。\nAction: str_replace_editor\nAction Input: {"command": "view", "path": "example.txt"}'
        elif "命令" in user_input or "bash" in user_input or "shell" in user_input:
            return 'Thought: 用户想执行命令。\nAction: bash\nAction Input: {"command": "ls -la"}'
        elif "grep" in user_input or "查找" in user_input:
            return 'Thought: 用户想搜索代码。\nAction: grep_search\nAction Input: {"query": "def main", "path": "."}'

        return "我理解了您的请求。这是一个模拟回复。"

    def _extract_city(self, text: str) -> str:
        match = re.search(r"([北京上海广州深圳杭州成都武汉西安]{2})", text)
        return match.group(1) if match else "北京"

    def _extract_expr(self, text: str) -> str:
        match = re.search(r"([\d\s+\-*/.()]+)", text)
        if match:
            expr = match.group(1).replace(" ", "")
            if expr and any(c in expr for c in "+-*/"):
                return expr
        match = re.search(r"(.+?)(?:等于|=|是多少)", text)
        if match:
            return match.group(1).strip().replace(" ", "")
        return "1+1"

    def _generate_summary(self, tool_result: str) -> str:
        if "get_weather" in tool_result:
            return "根据天气查询结果，今天天气不错，适合外出。"
        elif "calculate" in tool_result:
            return "计算已完成，结果如上所示。"
        elif "read_file" in tool_result:
            return "文件内容已读取，请查看上面的结果。"
        elif "query_database" in tool_result:
            return "数据库查询已完成，结果已返回。"
        elif "send_email" in tool_result:
            return "邮件已发送（模拟）。"
        elif "str_replace_editor" in tool_result:
            return "文件编辑已完成。"
        elif "bash" in tool_result:
            return "命令执行完成。"
        elif "grep_search" in tool_result:
            return "搜索完成，结果已返回。"
        return "任务已根据工具返回结果完成。"

class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                temperature=0.7,
                max_tokens=2000
            )
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_call = message.tool_calls[0]
                return f'Thought: 我需要调用工具。\nAction: {tool_call.function.name}\nAction Input: {tool_call.function.arguments}'
            return message.content or ""
        except Exception as e:
            return f"[LLM Error] {str(e)}"

    

