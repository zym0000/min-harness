"""
McpWrappedTool + discover_mcp_tools
适配 UnifiedToolRegistry.register_mcp_tools() 接口

接口契约 (register_mcp_tools 内部):
    tool = Tool(
        name=wt.name,
        description=wt.description,
        parameters=wt.parameters,       # List[ToolParameter]
        func=wt.execute,                # async (**kwargs) -> str
        tags=tags,                      # List[str]
        executor_type="async",
    )
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from tool_manger import ToolParameter
from MCP.mcp_bridge import McpClientPool

_LOG = logging.getLogger("harness.mcp_tools")


# 工具元数据配置 (可按需扩展或从外部配置加载)
# 工具名：标签
TOOL_TAGS: Dict[str, List[str]] = {
    "read_file":    ["general", "file_read"],
    "write_file":   ["file_write", "dangerous"],
    "apply_patch":  ["file_patch", "dangerous"],
    "list_dir":     ["general", "navigate"],
    "grep":         ["general", "search"],
    "search_code":  ["general", "search"],
    "shell_exec":   ["shell", "dangerous","git","test"],
    "file_info":    ["general", "file_read"],
}

APPROVAL_REQUIRED: Set[str] = {"write_file", "apply_patch", "shell_exec"}

# 类型映射：MCP JSON Schema type → ToolParameter type
_TYPE_MAP = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}

class McpWrappedTool:
    """
    包装单个 MCP 工具，暴露 UnifiedToolRegistry.register_mcp_tools 所需接口。

    Attributes:
        name:               工具名
        description:        工具描述
        parameters:         List[ToolParameter]
        tags:               标签列表（用于意图过滤）
        requires_approval:  是否需要审批
        server_name:        所属 MCP Server 名
    """

    def __init__(self,
                 name: str,
                 description: str,
                 input_schema: Dict[str, Any],
                 server_name: str,
                 pool: McpClientPool,
                 tags: Optional[List[str]] = None,
                 requires_approval: Optional[bool] = None):
        self.name = name
        self.description = description
        self.server_name = server_name
        self._pool = pool
        # 保留原始 schema，用于生成精确的 OpenAI function schema
        self._input_schema = self._normalize_schema(input_schema)
        self.parameters = self._parse_schema(self._input_schema)
        self.tags = tags or TOOL_TAGS.get(name, ["mcp"])
        self.requires_approval = (
            requires_approval
            if requires_approval is not None
            else name in APPROVAL_REQUIRED
        )

    @staticmethod
    def _normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """确保 schema 有基本结构，避免后续解析错误"""
        if not isinstance(schema, dict):
            schema = {}
        # 确保 type 存在且为 object
        if schema.get("type") != "object":
            schema["type"] = "object"
        # 确保 properties 存在且为 dict
        if "properties" not in schema or not isinstance(schema.get("properties"), dict):
            schema["properties"] = {}
        return schema

    def _parse_schema(self, schema: Dict[str, Any]) -> List[ToolParameter]:
        """将 MCP JSON Schema 转为 List[ToolParameter]，仅使用 name/type/description/required"""
        props = schema.get("properties", {})
        # 安全处理 required，防止 None
        required_list = schema.get("required") or []
        required_set = set(required_list)

        params: List[ToolParameter] = []
        for pname, pschema in props.items():
            if not isinstance(pschema, dict):
                pschema = {"type": "string"}

            raw_type = pschema.get("type", "string")
            param_type = _TYPE_MAP.get(raw_type, "string")

            # 只传递 ToolParameter 已知的通用字段，避免 enum/default 等不存在的属性
            params.append(ToolParameter(
                name=pname,
                type=param_type,
                description=pschema.get("description", ""),
                required=pname in required_set,
            ))

        return params

    async def execute(self, **kwargs) -> str:
        """
        通过 MCP Pool 调用远端工具。
        这是注册到 Tool.func 的实际执行体。
        """
        try:
            result = await self._pool.execute(
                self.server_name, self.name, kwargs
            )
            return result
        except ValueError as e:
            _LOG.error("MCP execute error [%s.%s]: %s",
                       self.server_name, self.name, e)
            return f"[ERROR] {e}"
        except asyncio.TimeoutError:
            _LOG.error("MCP execute timeout [%s.%s]", self.server_name, self.name)
            return f"[ERROR] Tool '{self.name}' timed out"
        except Exception as e:
            _LOG.exception("MCP execute unexpected error [%s.%s]",
                           self.server_name, self.name)
            return f"[ERROR] Tool '{self.name}' failed: {type(e).__name__}: {e}"

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        生成 OpenAI function calling schema。
        直接基于原始 input_schema，保证与 MCP 定义一致，
        不依赖 ToolParameter 的扩展属性。
        """
        props = self._input_schema.get("properties", {})
        required = self._input_schema.get("required") or []

        # 构建 OpenAI 所需的 properties 结构，保留 enum/default 等字段
        openai_props = {}
        for pname, pschema in props.items():
            if not isinstance(pschema, dict):
                pschema = {"type": "string"}
            prop_def = {"type": pschema.get("type", "string"),
                        "description": pschema.get("description", "")}
            # 传递可能存在的约束
            if "enum" in pschema:
                prop_def["enum"] = pschema["enum"]
            if "default" in pschema:
                prop_def["default"] = pschema["default"]
            openai_props[pname] = prop_def

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": openai_props,
                    "required": list(required),
                },
            },
        }

    def __repr__(self):
        ap = " " if self.requires_approval else ""
        return f"<McpWrappedTool:{self.name}{ap} [{self.server_name}]>"

async def discover_mcp_tools(
    pool: McpClientPool,
    *,
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
    timeout: float = 10.0,
    retries: int = 2,
) -> List[McpWrappedTool]:
    """
    从 McpClientPool 中发现所有已注册 Server 的工具，
    包装为 McpWrappedTool 列表，供 UnifiedToolRegistry.register_mcp_tools 使用。

    Args:
        pool:     已连接的 McpClientPool 实例
        include:  只保留这些工具名(one=全部）
        exclude:  排除这些工具名(None=不排除）
        timeout:  发现阶段的总超时（秒）
        retries:  失败重试次数

    Returns:
        List[McpWrappedTool] — 可直接传入 register_mcp_tools()

    Raises:
        RuntimeError: 发现失败（超时或所有重试耗尽）
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            schemas = await asyncio.wait_for(
                pool.get_all_tools(),
                timeout=timeout,
            )
            break
        except asyncio.TimeoutError:
            last_error = TimeoutError(
                f"MCP tool discovery timed out ({timeout}s), attempt {attempt}/{retries}"
            )
            _LOG.warning("discover_mcp_tools timeout (attempt %d/%d)", attempt, retries)
            if attempt < retries:
                await asyncio.sleep(0.5 * attempt)
        except Exception as e:
            last_error = e
            _LOG.warning("discover_mcp_tools error (attempt %d/%d): %s",
                         attempt, retries, e)
            if attempt < retries:
                await asyncio.sleep(0.5 * attempt)
    else:
        raise RuntimeError(
            f"Failed to discover MCP tools after {retries} attempts: {last_error}"
        )

    if include:
        schemas = [s for s in schemas if s.get("name") in include]
    if exclude:
        schemas = [s for s in schemas if s.get("name") not in exclude]

    tools: List[McpWrappedTool] = []
    for s in schemas:
        name = s.get("name", "")
        if not name:
            _LOG.warning("Skipping schema with empty name: %s", s)
            continue

        # 兼容不同字段名：parameters 或 inputSchema
        input_schema = s.get("parameters") or s.get("inputSchema") or {}
        if not isinstance(input_schema, dict):
            _LOG.warning("Tool '%s' has invalid input_schema, using empty", name)
            input_schema = {}

        tool = McpWrappedTool(
            name=name,
            description=s.get("description") or f"MCP tool: {name}",
            input_schema=input_schema,
            server_name=s.get("server_name", "unknown"),
            pool=pool,
        )
        tools.append(tool)
        _LOG.debug("  wrapped: %s (params=%d, tags=%s, approval=%s)",
                   name, len(tool.parameters), tool.tags, tool.requires_approval)

    _LOG.info("discover_mcp_tools: %d tools from %d schemas",
              len(tools), len(schemas))
    return tools

async def create_pool_and_discover(
    server_script: str,
    server_name: str = "coding-tools",
    mode: str = "InMemoryStack",
    mcp_variable_name: str = "mcp",
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
) -> tuple:
    """
    便捷入口：创建 McpClientPool → 连接 Server → 发现工具。

    Returns:
        (pool, tools) — pool 需要在 shutdown 时关闭
    """
    from MCP.mcp_bridge import McpServerConfig

    pool = McpClientPool()
    await pool.add_server(McpServerConfig(
        name=server_name,
        script_path=server_script,
        mode=mode,
        mcp_variable_name=mcp_variable_name,
    ))

    tools = await discover_mcp_tools(
        pool, include=include, exclude=exclude
    )
    return pool, tools