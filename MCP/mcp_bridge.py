import sys
import anyio
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack
from dataclasses import dataclass
from tool_manger import ToolParameter

from mcp import ClientSession
from mcp.server import Server
import uuid

@dataclass
class McpServerConfig:
    name:str
    script_path:str
    mode:str = "InMemoryStack"
    mcp_variable_name:str = "mcp"
    timeout:float = 30.0

class IntegratedMCPBridge:
    def __init__(self, 
                 mode: str = "InMemoryStack",
                 mcp_variable_name:str = "mcp",
                 timeout:float = 30.0
                 ):
        self.mode = mode
        self.mcp_variable_name = mcp_variable_name
        self.timeout = timeout
        self._server_name = "unknow"
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self._tg = None

    async def initialize(self, server_script_path: str):
        if self.mode == "InMemoryStack":
            await self._initialize_in_memory(server_script_path)
        elif self.mode == "OutofProcess":
            await self._init_out_of_process(server_script_path)
        else:
            raise ValueError(f"unknow mode:{self.mode}")

    async def _initialize_in_memory(self,server_script_path):
        print("mcp initialize memory")
        file_path = Path(server_script_path).resolve()
        module_name = f"_mcp_bridage_temp_{uuid.uuid4().hex[:8]}"

        spec = importlib.util.spec_from_file_location(module_name,file_path)
        if not spec or not spec.loader:
            raise ValueError("importlib spec file location failed")
        
        import_module = importlib.util.module_from_spec(spec)
        original_module = sys.modules.get(module_name)
        sys.modules[module_name] = import_module
        try:
            spec.loader.exec_module(import_module)
            local_fastmcp = getattr(import_module,self.mcp_variable_name,None)

            if local_fastmcp is None:
                raise ValueError(f"get fastmcp name:{self.mcp_variable_name} filed")
            
            if hasattr(local_fastmcp,"_mcp_server"):
                mcp_server_instance:Server = local_fastmcp._mcp_server
            elif hasattr(local_fastmcp,"server"):
                mcp_server_instance:Server = local_fastmcp.server
            else:
                mcp_server_instance = local_fastmcp
                
            self._server_name = getattr(local_fastmcp,'name','unknow')
            c2s_send, c2s_receive = anyio.create_memory_object_stream(100)
            s2c_send, s2c_receive = anyio.create_memory_object_stream(100)
            self._tg = await self.exit_stack.enter_async_context(anyio.create_task_group())

            async def _run_server():
                try:
                    await mcp_server_instance.run(
                        c2s_receive,s2c_send,
                        mcp_server_instance.create_initialization_options()
                    )
                except Exception as e:
                    print(f"[Bridge] Server crashed: {e}")
                    raise

            self._tg.start_soon(_run_server)
            #初始化 Client 并咬合管道
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(s2c_receive, c2s_send)
            )
            await self.session.initialize()

        finally:
            if original_module is not None:
                sys.modules[module_name] = original_module
            elif module_name in sys.modules:
                del sys.modules[module_name]
        
    async def _init_out_of_process(self,server_script_path):
            from mcp.client.stdio import stdio_client, StdioServerParameters
            server_params = StdioServerParameters(command="python", args=[server_script_path])
            read_stream, write_stream = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await self.session.initialize()

    async def fetch_tools_schema(self) -> List[Dict[str, Any]]:
        mcp_tools = await self.session.list_tools()
        #print(f"mcp_tools: { mcp_tools}")
        return [{"name": t.name, "description": t.description, "parameters": t.inputSchema} 
                for t in mcp_tools.tools]

    async def read_skill_guide_resource(self, skill_name: str) -> str:
        uri = f"skills://{skill_name}/guide"
        content_response = await self.session.read_resource(uri=uri)
        return " ".join([c.text for c in content_response.contents if hasattr(c, 'text')])

    async def execute_capability(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        result = await self.session.call_tool(name=tool_name, arguments=arguments)
        return "".join([c.text for c in result.content if hasattr(c, 'text')])

    async def shutdown(self):
        # 依赖 ExitStack 逆序关闭流，让 Server 自然退出
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
            except Exception as e:
                print(f"[Bridge] Error during shutdown: {e}")

class McpClientPool:
    def __init__(self):
        self.mcp_bridge:Dict[str,IntegratedMCPBridge] = {}
        self.mcp_config:Dict[str,McpServerConfig] = {}
    
    async def add_server(self, config:McpServerConfig):

        bridge = IntegratedMCPBridge(config.mode,
                                      config.mcp_variable_name,
                                      config.timeout)
        
        await bridge.initialize(config.script_path)
        
        self.mcp_bridge[config.name] = bridge
        self.mcp_config[config.name] = config

    async def get_all_tools(self):
        all_tools = []
        for name,bridage in self.mcp_bridge.items():
            tools = await bridage.fetch_tools_schema()
            for t in tools:
                t["server_name"] = name

            all_tools.extend(tools)

        return all_tools
    
    async def execute(self,server_name:str, tool_name:str,arguments: Dict[str, Any]):
        bridge = self.mcp_bridge.get(server_name,None)
        if not bridge:
            raise ValueError(f"server '{server_name}' not found in pool")
        return await bridge.execute_capability(tool_name,arguments)
    
    async def shutdown_all(self):
        for bridge in self.mcp_bridge.values():
            await bridge.shutdown()
        
        self.mcp_bridage.clear()
        self.mcp_config.clear()