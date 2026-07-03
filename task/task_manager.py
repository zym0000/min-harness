
import asyncio
import random
import time
from typing import Dict
from task.task_defined import TaskState,TaskStatus

class TaskManager:
    def __init__(self, max_history:int = 100):
        self.tasks: Dict[str,TaskState] = {}
        self.lock = asyncio.Lock()
        self.max_history = max_history

    async def create_task(self, user_input:str):
        task_id = f"{time.time() * 1000}_{random.randint(1000,9999)}"

        state = TaskState(
            task_id= task_id,
            user_input=user_input,
            task_status= TaskStatus.PENDING,
            current_step = 0,
            messages=[]
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
        
    async def get_all_task(self):
        async with self.lock:
            return dict(self.tasks)
    

        