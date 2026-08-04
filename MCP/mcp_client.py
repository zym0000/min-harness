from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Dict,List,Any

class MCPClientManager:
    """MCP 客户端管理器：管理外部 MCP 进程与会话生命周期"""
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        # 聚合管理多个 MCP 服务的会话 { "server_name": ClientSession }
        self.sessions: Dict[str, ClientSession] = {}

    async def connect_stdio_server(self, server_name: str, command: str, args: List[str]):
        """通过 Stdio 管道连接一个标准的 MCP 服务  例如 SQLite 或 Git MCP"""
        server_params = StdioServerParameters(command=command, args=args, env=None)

        #建立 stdio 管道
        read_stream, write_stream = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        #初始化标准 MCP 客户端会话
        session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        #触发 MCP 协议的标准握手生命周期
        await session.initialize()
        self.sessions[server_name] = session
        print(f"[MCP] 成功建立与远程服务 [{server_name}] 的标准协议握手")

    async def get_all_mcp_tools(self) -> List[Dict[str, Any]]:
        """获取所有已连入的 MCP 服务暴露的工具列表"""
        all_tools = []
        for server_name, session in self.sessions.items():
            # 调用官方协议中定义的方法
            mcp_response = await session.list_tools()
            for tool in mcp_response.tools:
                # 为了防止工具重名，加上命名空间前缀
                all_tools.append({
                    "mcp_server": server_name,
                    "name": f"{server_name}__{tool.name}",
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                    "dangerous": any(k in tool.name.lower() for k in ["delete", "write", "drop", "execute"])
                })
        return all_tools

    async def call_mcp_tool(self, namespaced_name: str, arguments: Dict[str, Any]) -> str:
        """触发执行远程 MCP 工具"""
        server_name, raw_tool_name = namespaced_name.split("__", 1)
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"未找到对应的 MCP 服务的活动会话: {server_name}")
        
        # 调用官方 SDK 执行远程 RPC Call
        result = await session.call_tool(name=raw_tool_name, arguments=arguments)
        
        # 解析标准 MCP 返回的 TextContent
        return "".join([content.text for content in result.content if hasattr(content, 'text')])

    async def close(self):
        """优雅关闭所有 MCP 连接与子进程占用的系统资源"""
        await self.exit_stack.aclose()