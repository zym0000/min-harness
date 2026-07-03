

import re
from tool_manger import ToolRegistry
from typing import List,Dict,Optional


class LLMClient:
    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None):
        return NotImplemented


class MockLLMClient(LLMClient):
    """
    模拟 LLM（无需 API Key）
    修复：正确识别工具结果消息，避免无限循环
    """
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        # 检查最后一条消息是否是工具结果 → 返回总结
        if messages and messages[-1].get("role") == "user":
            content = messages[-1].get("content", "")
            if "[Tool Result]" in content or "[Tool Error]" in content:
                return self._generate_summary(content)

        # 找到最后一条 user 消息
        user_input = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_input = msg.get("content", "").lower()
                break

        if "天气" in user_input or "weather" in user_input:
            city = self._extract_city(user_input)
            return f'Thought: 用户想查询天气，我需要调用天气工具。\nAction: get_weather\nAction Input: {{"city": "{city}"}}'
        elif "计算" in user_input or any(op in user_input for op in ["+", "-", "*", "/"]):
            expr = self._extract_expr(user_input)
            return f'Thought: 用户需要计算表达式。\nAction: calculate\nAction Input: {{"expression": "{expr}"}}'
        elif "文件" in user_input or "read" in user_input:
            path = self._extract_path(user_input)
            return f'Thought: 用户想读取文件。\nAction: read_file\nAction Input: {{"path": "{path}"}}'
        elif "数据库" in user_input or "query" in user_input or "搜索" in user_input:
            query = self._extract_query(user_input)
            return f'Thought: 用户需要查询数据库。\nAction: query_database\nAction Input: {{"query": "{query}"}}'
        elif "邮件" in user_input or "email" in user_input:
            return f'Thought: 用户想发送邮件，这是危险操作。\nAction: send_email\nAction Input: {{"to": "user@example.com", "subject": "测试", "body": "这是一封测试邮件"}}'

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

    def _extract_path(self, text: str) -> str:
        match = re.search(r"([\w./]+\.(txt|md|py|json))", text)
        return match.group(1) if match else "example.txt"

    def _extract_query(self, text: str) -> str:
        if "users" in text.lower() or "用户" in text:
            return "SELECT * FROM users"
        if "orders" in text.lower() or "订单" in text:
            return "SELECT * FROM orders"
        return "SELECT * FROM users"

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

    

