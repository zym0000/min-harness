# main.py
import asyncio
from mcp_bridge import IntegratedMCPBridge

async def main():
    print("=== 启动多工具 Agent 测试 ===\n")
    bridge = IntegratedMCPBridge(mode="InMemoryStack")
    await bridge.initialize("test/multi_tool_server.py")
    
    print("--- 阶段 1: 发现所有可用工具 ---")
    tools = await bridge.fetch_tools_schema()
    for tool in tools:
        print(f"[{tool['name']}] - {tool['description']}")
    
    print("\n--- 场景 A: 用户问 '北京天气怎么样？' ---")
    print("Agent 思考: 我需要查天气，但我不知道城市名该传中文还是拼音，先读指南...")
    
    # 1. 获取天气指南 (渐进式披露)
    weather_guide = await bridge.read_skill_guide_resource("weather")
    print(f"指南提示: {weather_guide.strip().split(chr(10))[2]}") # 提取关键避坑点
    
    # 2. 根据指南精准调用
    print("Agent 思考: 指南说必须用拼音，所以我传 'Beijing'。")
    result = await bridge.execute_capability("get_weather", {"city": "Beijing"})
    print(f"执行结果: {result}")

    print("\n--- 场景 B: 用户说 '给老板发封邮件汇报进度' ---")
    print("Agent 思考: 发邮件是高危操作，我得看看有什么注意事项...")
    
    # 1. 获取邮件指南
    email_guide = await bridge.read_skill_guide_resource("email")
    print(f"指南警告: {email_guide.strip().split(chr(10))[2]}") 
    
    # 2. 模拟 Agent 的审批逻辑 (结合你 Harness 中的 ApprovalGate)
    print("Agent 动作: 触发 ApprovalGate，等待人类确认邮件内容...")
    # (此处省略人类点击确认的代码)
    
    # 3. 确认后调用
    result = await bridge.execute_capability(
        "send_email", 
        {"to": "boss@company.com", "subject": "进度汇报", "body": "项目已完成 80%。"}
    )
    print(f"执行结果: {result}")
    
    await bridge.shutdown()
    print("\n=== 测试结束 ===")

if __name__ == "__main__":
    asyncio.run(main())