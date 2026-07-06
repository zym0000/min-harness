import re
import json
from dataclasses import dataclass
from typing import Any, Dict,Optional,List
from tool_execute_process.error_classify import ToolErrorType

class ToolParameter:
    def __init__(self, name, type, description, required = False):
        self.name = name
        self.type = type
        self.description = description
        self.required = required #参数是否是必须的

    def to_dict(self):
        return {
            "name":self.name,
            "type":self.type,
            "description":self.description,
            "required":self.required
        }

class Tool:
    def __init__(self,name,
            description,
            parameters,
            func,
            tags,
            dangerous = False,
            executor_type:str  = "async"):
        
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters
        self.dangerous = dangerous #函数执行危险标记
        self.tags = tags
        self.executor_type = executor_type, #async, subprocess


    #大模型tool 调用有两种方式一个React 模板
    #另外一种是大模型本身支持API tool_call方式
    #这里为了实现主流的，两种方式都实现,下面的实现，都是给LLM 提示词

    #react格式如下:
    # You have access to the following tools:

    # get_weather: 获取指定城市的实时天气信息。
    # Parameters:
    #     - city (string): 城市名称，使用标准中文名，如'北京'。
    #     - unit (string): 温度单位，支持 'celsius' 或 'fahrenheit'，默认celsius。

    # To use a tool, respond with:
    # Thought: ...
    # Action: get_weather[city="北京", unit="celsius"]
    
    def to_react_description(self):
        parameter_doc = []

        for p in self.parameters:
            req = "required" if p.required else "optional"
            parameter_doc.append(f" - {p.name} ({p.type}, {req}): {p.description}")
        
        parameter_block = "\n".join(parameter_doc) if parameter_doc else "  parameterless"

        return f"""{self.name}: {self.description}
        parameters: {parameter_block}
        """
    
    # {
    #     "name":"get_weather",
    #     "description":"获取指定城市的实时天气信息",
    #     "parameters":{
    #         "type":"object",
    #         "properties":properties,
    #         "required":required
    #     }
    # }
    #这里是以open ai 样式进行写的 tool_call，可以增加接口，实现自己模型相关方式
    def to_openai_schema(self):
        required = []
        properties ={}
        for p in self.parameters:
            properties[p.name] = {
                "type":p.type,
                "description":p.description
            }

            if p.required:
                required.append(p.name)

        return{
            "type":"function",
            "function":{
                "name":self.name,
                "description":self.description,
                "parameters":{
                    "type":"object",
                    "properties":properties,
                    "required":required
                }
            }
        }
    
    async def execute(self, **kwargs):
        for param in self.parameters:
            param_dict = param.to_dict()
            if param_dict.get('required') and param_dict.get("name") not in kwargs:
                raise ValueError(f"缺少必要参数 {param_dict.get("name")}")
        
        return await self.func(**kwargs)

@dataclass
class ToolCall:
    '''
        Structured tool invocation request
    '''
    tool_name:str
    arguments:Dict[str,Any]
    raw_text:str
    thought:str

class ToolCallParser:

    #解析LLM 输出
    @staticmethod
    def parse(text: str):
        text = text.strip()
        thought = ""
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|$)", text, re.DOTALL | re.IGNORECASE
        )

        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        if not action_match:
            return None, None

        tool_name = action_match.group(1)
        input_match = re.search(
            r"Action Input:\s*(\{.*?\})", text, re.DOTALL | re.IGNORECASE
        )

        if input_match:
            try:
                arguments = json.loads(input_match.group(1))
                return (
                    ToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                        raw_text=text,
                        thought=thought,
                    ),
                    None,
                )
            except json.JSONDecodeError as e:
                return None, f"Action Input JSON 解析失败: {e}"

        return (
            ToolCall(tool_name=tool_name, arguments={}, raw_text=text, thought=thought),
            None,
        )

    @staticmethod
    def _parse_json_format(text: str, thought: str) -> Optional[ToolCall]:
        action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        if not action_match:
            return None

        tool_name = action_match.group(1)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL | re.IGNORECASE)

        if input_match:
            try:
                arguments = json.loads(input_match.group(1))
                return ToolCall(tool_name=tool_name, arguments=arguments, raw_text=text, thought=thought)
            except json.JSONDecodeError:
                pass

        return ToolCall(tool_name=tool_name, arguments={}, raw_text=text, thought=thought)

    @staticmethod
    def _parse_bracket_format(text: str, thought: str) -> Optional[ToolCall]:
        bracket_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", text, re.DOTALL | re.IGNORECASE)
        if not bracket_match:
            return None

        tool_name = bracket_match.group(1)
        args_str = bracket_match.group(2)

        arguments = {}
        # 匹配 key="val" 或 key='val' 或 key=val
        pattern = r'(\w+)=\s*["\']?(.*?)["\']?(?:,\s*|\s*$)'
        for pair in re.findall(pattern, args_str):
            key, val = pair[0], pair[1].strip().strip('"').strip("'")
            arguments[key] = val

        return ToolCall(tool_name=tool_name, arguments=arguments, raw_text=text, thought=thought)

    @staticmethod
    def _parse_code_block(text: str, thought: str) -> Optional[ToolCall]:
        code_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if not code_match:
            return None
        try:
            data = json.loads(code_match.group(1))
            if "tool" in data or "name" in data:
                return ToolCall(
                    tool_name=data.get("tool") or data.get("name"),
                    arguments=data.get("arguments") or data.get("params") or {},
                    raw_text=text,
                    thought=thought
                )
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def is_tool_call(text: str) -> bool:
        return ToolCallParser.parse(text) is not None

@dataclass
class ToolResult:
    '''
        工具执行结果
    '''
    tool_name:str
    arguments:Dict[str,Any]
    output:Any
    error:Optional[str] = None
    retry_count:int = 0
    error_type:Optional[ToolErrorType] = None
    is_retryable:bool = False

    @property
    def is_error(self):
        return self.error if not None else ""
    
    #这里返回的只是个人采用的格式，这里可以自己定义返回格式类型
    def to_text(self):
        if self.error:
            return f"[Tool Error] {self.tool_name}: {self.error}"
        return f"[Tool Result] {self.tool_name}:{self.output}"

class ToolRegistry:
    '''
        工具管理
    '''

    def __init__(self):
        self.tools : Dict[str,Tool]= {}

    def register(self, tool:Tool):
        if tool.name in self.tools:
            raise ValueError(f"tool:{tool.name} is registered")
        self.tools[tool.name] = tool

    def get(self,name):
        if name not in self.tools:
            raise KeyError(f"get tool name:{name} not find")
        return self.tools.get(name)
    
    def list_tools(self):
        return list(self.tools.values())
    
    def validate_call(self,call:ToolCall):
        try:
            tool = self.get(call.tool_name)
        except KeyError as e:
            return str(e)
        
        for param in tool.parameters:
            param_dict = param.to_dict()
            if param_dict.get("required") and param_dict.get("name") not in call.arguments:
                return f"missing required parame:{param_dict.get("name")}"
        return None
    
    def describe_tools(self, tools: Optional[List[Tool]] = None) -> str:
        tool_list = tools or self.list_tools()
        return "\n\n".join([t.to_react_description() for t in tool_list])

    def to_openai_sechme(self,tools:Optional[List[Tool]] = None):
        if tools is None:
            return []
        
        return [t.to_openai_schema() for t in tools]
    
    def filter_by_tags(self, tags: List[str]) -> List[Tool]:
        if not tags:
            return []
        
        return [t for t in self.tools.values() if any(tag in t.tags for tag in tags)]

    

    

    
        



        

        