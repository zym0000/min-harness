from dataclasses import dataclass
from enum import auto
from typing import Optional,List,Dict
from tool_manger import ToolCall,ToolResult,ToolRegistry,ToolCallParser,Tool
from llm_client import LLMClient
from datetime import datetime
from tool_execute_process.retry_engine import RetryEngine
from tool_execute_process.tool_metrics import MetricsControll
from tool_execute_process.error_classify import LLMErrorClassify
from task.task_manager import TaskManager,TaskStatus
from async_execution_engine import AsyncExecutionEngine
from event.event import LoopEvent,EventType

import time
import asyncio

class LoopState:
    '''
        agent loop state
    '''
    IDIL = auto()
    THIKNING = auto()
    PARSING = auto()
    VALIDATING = auto()
    ACTIVE = auto()
    OBSERVING = auto()
    FEEDBACK = auto()
    FINISHED = auto()
    MAX_STEPS_REACHED = auto()
    ERROR = auto()


@dataclass
class StepRecord:
    step_num:int
    state:LoopState
    timestamp:str
    llm_out:str = ""
    tool_call:Optional[ToolCall] = None
    tool_result:Optional[ToolResult] = None
    final_answer:str = ""

@dataclass
class RunResult:
    input:str
    final_answer:str
    total_steps:int
    steps:List[StepRecord]
    final_state:LoopState
    execution_time_ms:float=0.0
    total_cost_estimate:float = 0.0

class AgentLoop:
    def __init__(self, tool_registry:ToolRegistry, 
                 llm:LLMClient, 
                 task_manager: TaskManager,
                 execution_engine:AsyncExecutionEngine,
                 cancel_event: asyncio.Event,
                 max_steps = 10,
                 retry_engine:Optional[RetryEngine] = None):
        
        self.tool_registry = tool_registry
        self.llm = llm
        self.cancel_event = cancel_event
        self.max_step = max_steps
        self.parser = ToolCallParser()
        self.retry_engine = RetryEngine(retry_engine)
        self.execute_engine = execution_engine
        self.task_manager = task_manager

    async def _check_cancelled(self):
        if self.cancel_event.is_set():
            raise asyncio.CancelledError()

    async def run(self,task_id,user_input:str,tools:List[Tool]):
        final_status = TaskStatus.FAILED

        await self.task_manager.update_state(
            task_id,
            status=TaskStatus.RUNNING,
            current_step=0,
            messages=[
                {"role": "user", "content": user_input},
            ],
        )

        yield LoopEvent(
            event_type= EventType.TASK_STARTED,
            task_id=task_id,
            timestamp= time.time(),
            content=user_input,
        )

        try:
            async for event in self._execute_steps(task_id,tools):
                yield event
        except asyncio.CancelledError:
            final_status = TaskStatus.CANCELLED
            yield LoopEvent(
                event_type=EventType.TASK_CANCELLED,
                task_id=task_id,
                timestamp=time.time()
            )
        except Exception as e:
            final_status = TaskStatus.FAILED
            yield LoopEvent(
                event_type= EventType.TASK_FAILED,
                task_id= task_id,
                timestamp=time.time(),
                content = str(e)
            )
            raise
        finally:
            #保证UI或上层突然中断，可以正确完成
            state = await self.task_manager.get_state(task_id)
            if state is not None and state.task_status == TaskStatus.RUNNING and final_status is not None:
                await self.task_manager.update_state(task_id=task_id,status = final_status)
    
    async def _execute_steps(self,task_id,tools:List[Tool]):
        for step_num in range(1,self.max_step+ 1):
            await self.task_manager.update_state(task_id=task_id,current_step = step_num)
            await self._check_cancelled()

            state = await self.task_manager.get_state(task_id)

            messages = list(state.messages) if state else []

            yield LoopEvent(
                event_type=EventType.THINKNING_STARTED,
                task_id=task_id,
                timestamp= time.time(),
                step_num=step_num
            )

            tool_schema  = self.tool_registry.to_openai_sechme(tools)
            llm_start_time = time.time()
            llm_output = self.llm.chat(messages,tool_schema)
            llm_latency = (time.time()-llm_start_time)*1000

            yield LoopEvent(
                event_type=EventType.THINKING_COMPLETED,
                task_id=task_id,
                timestamp=time.time(),
                step_num=step_num,
                content=llm_output,
                data={"llm_latency_ms": round(llm_latency, 2)},
            )

            tool_call, parse_error= self.parser.parse(llm_output)
            if parse_error:
                yield LoopEvent(
                    event_type=EventType.TOOL_VALIDATION_FAILED,
                    task_id=task_id,
                    timestamp=time.time(),
                    step_num=step_num,
                    content=parse_error
                )

                new_message = messages + [
                    {"role":"assistant","content":llm_output},
                    {"role":"user","content":f"Observation [Parse Error]:{parse_error}"}
                ]

                await self.task_manager.update_state(task_id=task_id,messages=new_message)
                continue

            if tool_call is None:
                yield LoopEvent(
                    event_type=EventType.FINAL_ANSWER,
                    task_id=task_id,
                    timestamp=time.time(),
                    step_num= step_num,
                    content=llm_output
                )

                yield LoopEvent(
                    event_type=EventType.TASK_COMPLETED,
                    task_id=task_id,
                    timestamp=time.time()
                )

                await self.task_manager.update_state(task_id,task_status=TaskStatus.COMPLETED)
                return
            
            yield LoopEvent(
                event_type=EventType.TOOL_CALL_PARSED,
                task_id=task_id,
                timestamp=time.time(),
                step_num=step_num,
                tool_name=tool_call.tool_name,
                data={
                    "arguments":tool_call.arguments
                }
            )

            validate_error = self.tool_registry.validate_call(tool_call)
            if validate_error:
                yield LoopEvent(
                    event_type=EventType.TOOL_VALIDATION_FAILED,
                    task_id = task_id,
                    timestamp=time.time(),
                    step_num=step_num,
                    content=validate_error
                )

                new_message = messages + [
                    {"role":"assistant","content":llm_output},
                    {"role":"user","content":f"Observation [Tool Error]: {validate_error}"}
                ]

                await self.task_manager.update_state(task_id,messages=new_message)

                continue

            yield LoopEvent(
                event_type=EventType.TOOL_VALIDATION_PASSED,
                task_id=task_id,
                timestamp=time.time(),
                step_num=step_num,
                tool_name=tool_call.tool_name
            )

            tool = self.tool_registry.get(tool_call.tool_name)

            yield LoopEvent(
                event_type=EventType.TOOL_EXECUTION_STARTED,
                task_id=task_id,
                tool_name=tool.name,
                timestamp=time.time(),
                data={"arguments":tool_call.arguments}
            )

            result, tool_latency_ms = await self._execute_with_retry(task_id, tool,tool_call)

            if result.is_error:
                yield LoopEvent(
                    event_type=EventType.TOOL_EXECUTION_FAILED,
                    task_id = task_id,
                    tool_name=tool.name,
                    error=result.error,
                    data ={
                        "error_type":result.error_type,
                        "retry_count": result.retry_count,
                        "latency_ms": round(tool_latency_ms, 2),
                    }
                )
            
            yield LoopEvent(
                event_type=EventType.TOOL_EXECUTION_COMPLETED,
                task_id=task_id,
                timestamp=time.time(),
                tool_name=tool.name,
                data={
                    "latency_ms": round(tool_latency_ms, 2),
                    "retry_count": result.retry_count,
                },
            )
            
            new_messages = messages + [
                    {"role":"assistant","content":llm_output},
                    {"role":"user","content":f"Observation:{result.to_text()}"}
                ]

            await self.task_manager.update_state(task_id, messages = new_messages)

        yield LoopEvent(
            event_type=EventType.TASK_FAILED,
            task_id=task_id,
            timestamp=time.time(),
            content="达到最大步数限制",
        )
        await self.task_manager.update_state(task_id, status=TaskStatus.FAILED)


    async def _execute_with_retry(self,task_id,tool:Tool,call:ToolCall):
        last_result = None
        total_latency_ms = 0.0

        for attempt in range(self.retry_engine.policy.max_retries + 1):
            await self._check_cancelled()

            start_time = time.time()
            result = await self.execute_engine.execute(task_id,tool,call.arguments)
            total_latency_ms += start_time

            if not result.is_error :
                result.retry_count = attempt
                return result,total_latency_ms
            
            last_result = result
            
            if attempt < self.retry_engine.policy.max_retries and result.is_retryable:
                delay = self.retry_engine.calc_delay(attempt,result.error_type)
                # yield LoopEvent(
                #     event_type=EventType.TOOL_RETRY_SCHEDULED,
                #     task_id=task_id,
                #     tool_name= tool.name,
                #     timestamp=time.time(),
                #     data={
                #         "attempt":attempt+1,
                #         "delay":delay
                #     }
                # )   

                await asyncio.sleep(delay)
                result.retry_count = attempt + 1
            else:
                break

            return last_result,total_latency_ms


    def _execute_with_defense(self,tool:Tool, call:ToolCall):
        
        result,error_msg,retry_count = self.retry_engine.execute(
            tool.execute, **call.arguments)
        
        if error_msg:
            error_type = LLMErrorClassify.classify(error_msg)
            return ToolResult(
                tool_name= call.tool_name,
                arguments= call.arguments,
                output= None,
                error= error_msg,
                error_type= error_type,
                retry_count = retry_count
            )

        return ToolResult(
            tool_name=call.tool_name,
            arguments=call.arguments,
            output=result,
            retry_count= retry_count
        )

        







