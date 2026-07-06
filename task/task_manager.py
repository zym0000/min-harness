
import asyncio
import random
import time
from typing import Dict,List,Optional,Any
from task.task_defined import TaskState,TaskStatus,ContinueResult
from context_message import ContextManager

class TaskManager:
    def __init__(self, max_history:int = 100):
        self.tasks: Dict[str,TaskState] = {}
        self.lock = asyncio.Lock()
        self.max_history = max_history

    async def create_task(self, user_input:str,system_prompt:str):
        task_id = f"{time.time() * 1000}_{random.randint(1000,9999)}"

        state = TaskState(
            task_id= task_id,
            user_input=user_input,
            task_status= TaskStatus.PENDING,
            current_step = 0,
            messages=[{"role": "user", "content": user_input}],
            system_prompt = system_prompt
        )

        async with self.lock:
            self.tasks[task_id] = state

        return task_id
    
    async def get_state(self,task_id):
        async with self.lock:
            return self.tasks.get(task_id)
    
    async def register_task(self,task_id,async_task:asyncio.Task):
        async with self.lock:
            self.async_task[task_id] = async_task

    async def update_state(self,task_id, **kwargs):
        async with self.lock:
            if task_id in self.tasks:
                for key, value in kwargs.items():
                    setattr(self.tasks[task_id], key,value)
                self.tasks[task_id].updated_at = time.time()
    
    async def delete_task(self, task_id):
        async with self.lock:
            return self.tasks.pop(task_id,None) is not None
        
    async def get_all_tasks(self):
        async with self.lock:
            return dict(self.tasks)
        
    async def append_msg(self, task_id: str, new_msgs: List[Dict[str, str]]):
        async with self.lock:
            state:TaskState =self.tasks[task_id]
            if not state:
                return
            state.messages.extend(new_msgs)
            state.updated_at = time.time()

    async def compress_messages(self,
                                task_id:str,
                                llm:Optional[Any],
                                context_message:ContextManager):
        
        async with self.lock:
           state:TaskState = self.tasks[task_id]
           
           if not state:
               return []
           
           system_prompt=state.system_prompt
           history_message = list(state.messages)
           history_len = len(history_message)
           task_state = state

        messages = await context_message.prepare_message(
            system_prompt=system_prompt,
            history=history_message,
            llm_client=llm,task_state=task_state)
        
        if messages and messages[0]["role"] == "system":
            actual_system = messages[0]
            compressed_history = messages[1:]  # 剩余为非 system 消息
        else:
            # 如果 prepare_messages 异常丢失 system，回退旧值
            actual_system = {"role": "system", "content": system_prompt}
            compressed_history = [m for m in messages if m["role"] != "system"]
    
        async with self.lock:
            if task_id  not in self.tasks:
                return []
                   
            current = self.tasks[task_id].messages

            if len(current) > history_len:
                new_message = current[history_len:]
                compressed_history = compressed_history + new_message
            
            self.tasks[task_id].messages = compressed_history
            self.tasks[task_id].updated_at = time.time()

            return [actual_system] + compressed_history
    
    async def drain_pending_input(self, task_id:str):
        async with self.lock:
            state:TaskState = self.tasks[task_id]
            if not state or not state.pending_input:
                return []
            
            inputs = list(state.pending_input)
            state.pending_input.clear()
            return inputs
        
    async def continue_task(self,task_id:str, user_input:str):
        async with self.lock:
            if task_id not in self.tasks:
                return ContinueResult.FAILED
            
            state:TaskState = self.tasks[task_id]

            if state.task_status in {TaskStatus.COMPLETED,TaskStatus.FAILED,TaskStatus.CANCELLED}:
                state.task_status = TaskStatus.RUNNING
                state.messages.append({"role": "user", "content": user_input})
                state.updated_at = time.time()
                return ContinueResult.ACTIVATED
            
            if state.task_status in {TaskStatus.RUNNING,TaskStatus.PAUSED}:
                state.pending_input.append(user_input)
                state.updated_at = time.time()
                return ContinueResult.QUEUED
            
            return ContinueResult.FAILED


            

        