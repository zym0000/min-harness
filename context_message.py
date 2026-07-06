from token_estimate import TokenEstimate
from typing import Dict,Optional
from llm_client import LLMClient
from enum import Enum, auto
from task.task_manager import TaskState
import re

class CompressionStrategy(Enum):
    PRESERVE_ALL = auto()   # 不压缩，保留全部（短对话/调试）
    WINDOW = auto()         # 滑动窗口，保留最近 N 轮
    TASK_AWARE = auto()     # 任务感知压缩

class ContextManager:
    def __init__(self,
            max_tokens= 8000,  #LLM 窗口大小
            reserve_tokens = 2000, #预留给LLM生成回复的token 大小
            keep_recent_turns = 3,
            strategy:CompressionStrategy = CompressionStrategy.TASK_AWARE): #保留最近的轮数
        
        self.max_tokens = max_tokens
        self.resver = reserve_tokens
        self.recent_turns = keep_recent_turns * 2
        self.token_estimate = TokenEstimate()
        self.strategy = strategy

    async def prepare_message(self,system_prompt:str,
                              history:list[Dict[str,str]],
                              llm_client:Optional[LLMClient] = None,
                              task_state:Optional[TaskState] = None):
        
        if self.strategy == CompressionStrategy.TASK_AWARE and task_state:
            messages = await self._build_aware(system_prompt, history,llm_client,task_state)
        elif self.strategy == CompressionStrategy.WINDOW:
            messages = self._build_window(system_prompt,history)
        else:
            messages = self._build_preserve_all(system_prompt,history)
        
        #结构归一化：确保唯一 system 且在最前
        messages = self._ensure_single_system_front(messages)
        #判断token 是否超过最大token,进行压缩
        messages = self._emergency_truncate(messages)
        #确保角色是交替进行的system->user->assistant->user......
        messages  = self._merge_consecutive_roles(messages)

        return messages

    def _build_preserve_all(self,system_prompt:str, history:list[Dict[str,str]]):
        messages = [{"role":"system","content":system_prompt}] + history
        return messages
    
    def _build_window(self,system_prompt:str,history:list[Dict[str,str]]):
        recent_turns = history[:-self.recent_turns] if len(history) > self.recent_turns else history
        messaages = [{"role":"system","content":system_prompt}] + recent_turns
        return messaages
    
    async def _build_aware(self,system_prompt:str,
                     history:list[Dict[str,str]],
                     llm_client:Optional[LLMClient],
                     task_state:Optional[TaskState]):
        
        if len(history) > self.recent_turns:
            recent = history[-self.recent_turns:]
            old = history[:-self.recent_turns]
        else:
            old = []
            recent = history

        if old and llm_client:
            new_memroy = await self._extract_task_memory(old, llm_client,task_state)

            if new_memroy:
                task_state.memory_segment = new_memroy

        system_parts  = [system_prompt]
        if task_state.task_summary:
            system_parts.append(f"[Task goal]: {task_state.task_summary}")
        
        if task_state.key_facts:
            facts_text = "\n\n".join(f"-{f}" for f in task_state.key_facts)
            system_parts.append(f"[key facts]\n{facts_text}")

        system = "\n\n".join(system_parts)

        messages = [{"role":"system","content":system}]

        if task_state.memory_segment:
            messages.append({
                "role":"user",
                "content":task_state.memory_segment
            })

        messages.extend(recent)
        return messages
    
    async def _extract_task_memory(self,old_message:list[Dict[str,str]],
                                    llm_client:Optional[LLMClient],
                                    task_state: TaskState):
        
        #这里提取10条，会有迷惑，如果我们想一个任务流执行
        #从1->100, 那么100输出的信息是会包含前面的信息的
        #我们看整个流程，由1得出2 2的回答是大模型由1信息回答，以此，后面的信息是根据前面
        #所以这里取10条足够了
        history_text = ""
        for msg in old_message[-10:]:
            role = msg.get("role","UNKNOW")
            content = msg.get("content")[:500] #这里取500 可能会对语义造成影响，这里可以换成头...尾的提取方式
            history_text += f"{role}:{content}\n"
        
        print(f"")
        prompt = f"""请根据以下对话历史，提取任务进展信息。只输出指定格式，不要解释。
        <history>
        {history_text}
        </history>

        <current_task_summary>
        {task_state.task_summary or "None"}
        </current_task_summary>

        <current_key_facts>
        {chr(10).join(f"- {f}" for f in task_state.key_facts) or "None"}
        </current_key_facts>
        
        请输出(保持简洁,200字以内):
        任务目标:[一句话概括用户最终想达成什么，如果已有则保持一致]
        已完成:[已完成的关键步骤]
        待处理:[还需要做的]
        关键发现:[重要事实、数据、决策]
        输出: 
        """
        try:
            result = await llm_client.chat(
                [{"role": "user", "content": prompt}],tools=None)
            
            self._parse_and_update_state(result, task_state)

            return result.strip()
        except Exception:
            return None
        
    def _parse_and_update_state(self,memory_text:str,task_state:TaskState):
        goal_match  = re.search(r"任务目标[：:]\s*(.+)", memory_text)
        if goal_match:
            task_state.task_summary = goal_match.groups(1).strip()
        
        for line in memory_text.split("\n"):
            line = line.strip()
            if line.startswith('- ') and len(line) > 3:
                fact = line[2:].strip()
                if fact and fact not in task_state.key_facts:
                    task_state.key_facts.append(fact)
                    #防止存储爆炸
                    if len(task_state.key_facts) > 20:
                        task_state.key_facts.pop(0)

    @staticmethod
    def _merge_consecutive_roles(messages:list[Dict[str,str]]):
        if not messages:
            return messages
        
        #system消息
        merged = [messages[0]]
        #把相邻 role的角色进行合并，避免LLM 错误
        for msg in messages[1:]:
            prve = merged[-1] #最后一条
            if msg["role"] == prve["role"] and msg["role"] != "system":
                prve["content"] += "\n" + msg["content"]
            else:
                merged.append(msg)

        return merged
        
    #确保 所有 system在顶部,并且保证一个message 只有一个system
    @staticmethod
    def _ensure_single_system_front(messages:list[Dict[str,str]]):
        system_contents = []
        non_system = []
        for message in messages:
            if message["role"] == "system":
                system_contents.append(message["content"])
            else:
                non_system.append(message)
        
        if not system_contents:
            return non_system
        
        combined_system = "\n\n".join(system_contents)
        return [{"role":"system","content":combined_system}] + non_system
    
    def _emergency_truncate(self,messages:list[Dict[str,str]]):
        budget = self.max_tokens - self.resver
        current_total = self.token_estimate.estimate_message(messages)

        #system + 最近两轮对话就是 size*2 2(4)
        if budget <= current_total or len(messages) <=5:
            return messages
        
        preserved_head = [messages[0]]
        preserved_tail = messages[-4:]
        middle = messages[1:-4]

        while middle and current_total > budget:
            discarded_msg = middle.pop(0)
            discarded_tokens = self.token_estimate.estimate(discarded_msg.get("content", "")) + 4
            current_total -= discarded_tokens

        result = preserved_head + middle + preserved_tail

        #最后兜底，直接把早期的中间的内容进行合并
        if self.token_estimate.estimate_message(result) > budget and middle:
            truncated_middle = "[Earlier context truncated due to length limit]"
            result = preserved_head + [{"role":"user","content":truncated_middle}] + preserved_tail

        return result

        

        