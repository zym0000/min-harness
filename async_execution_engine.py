import asyncio
from typing import Dict,Any
from tool_manger import Tool,ToolResult,ToolErrorType
from tool_execute_process.error_classify import LLMErrorClassify
import traceback

class AsyncExecutionEngine:
    def __init__(self):
        self.run_task:Dict[str,asyncio.Task] = {}

    async def execute(self, task_id, 
                      tool:Tool, 
                      arguments:Dict[str,Any],timeout = 30.0):
        
        async def _runner():
            try:
                if tool.executor_type == "subprocess":
                    await self._excute_subprocess(tool,arguments)
                else:
                    output = await tool.execute(task_id,**arguments)
                    return ToolResult(
                        tool_name= tool.name,
                        arguments=arguments,
                        output= output
                    )
            except Exception as e:
                traceback.print_exc()
                error_type = LLMErrorClassify.classify(str(e))
                return ToolResult(
                    tool_name= tool.name,
                    arguments=arguments,
                    output=None,
                    error = str(e),
                    error_type= error_type,
                    is_retryable= LLMErrorClassify.is_retryable(error_type)
                )
        
        task = asyncio.create_task(_runner())
        self.run_task[task_id] = task

        try:
            result = await asyncio.wait_for(task,timeout)
            return result
        except asyncio.TimeoutError:
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            return ToolResult(
                tool_name= tool.name,
                arguments=arguments,
                output=None,
                error = "excute timeout",
                error_type=ToolErrorType.NETWORK_TIMEOUT,
                is_retryable = True
            )
        finally:
            self.run_task.pop(task_id,None)
    
    async def _excute_subprocess(self,tool:Tool,arguments:Dict[str,Any]):
        pass

    async def cancel_execution(self,task_id:str):
        task = self.run_task.get(task_id,None)
        if task:
            task.cancel()
            try:
                await asyncio.wait_for(task, 1.0)
            except (asyncio.TimeoutError,asyncio.CancelledError):
                pass

    async def shutdown(self) -> None:
        """退出时取消所有仍在执行的工具任务并等待回收。"""
        if not self.run_task:
            return
        tasks = list(self.run_task.values())
        self.run_task.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


        
