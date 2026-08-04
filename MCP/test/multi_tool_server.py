# multi_tool_server.py
from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP 实例
mcp = FastMCP("OmniSkillServer")

# ================= 工具区 (Tools) =================

@mcp.tool()
def calculate_product(a: int, b: int) -> str:
    """计算两个整数的乘积"""
    return f"计算成功: {a} * {b} = {a * b}"

@mcp.tool()
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的实时天气"""
    # 模拟 API 返回
    return f"{city} 当前天气晴朗，温度 26° {unit}。"

@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """发送电子邮件 (危险操作，需要谨慎)"""
    return f"邮件已成功发送至 {to}，主题: {subject}"

# ================= 指南区 (Resources for 渐进式披露) =================

@mcp.resource("skills://math/guide")
def math_guide() -> str:
    return """
    # 数学计算指南
    - 仅支持整数乘法，严禁传入浮点数或字符串。
    - 参数: `a` (乘数), `b` (被乘数)。
    """

@mcp.resource("skills://weather/guide")
def weather_guide() -> str:
    return """
    # 天气查询指南
    - `city`: 必须使用标准拼音或英文（如 'Beijing', 'Shanghai'），不要传中文！
    - `unit`: 可选 'celsius' 或 'fahrenheit'，默认摄氏度。
    """

@mcp.resource("skills://email/guide")
def email_guide() -> str:
    return """
    # 邮件发送避坑指南
    - 这是一个【高权限操作】，调用前务必向用户确认收件人和内容！
    - `body` 支持 Markdown 格式。
    """

if __name__ == "__main__":
    mcp.run()