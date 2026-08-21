from token_estimate import TokenEstimate
from typing import Dict, Optional, List, Tuple
from llm_client import LLMClient
from enum import Enum, auto
from task.task_defined import TaskState
import asyncio
import logging
import re
import textwrap

_LOG = logging.getLogger("harness.context")

class CompressionStrategy(Enum):
    PRESERVE_ALL = auto()   # 不压缩，保留全部（短对话/调试）
    WINDOW = auto()         # 滑动窗口，保留最近 N 轮
    TASK_AWARE = auto()     # 任务感知压缩

class ContextManager:

    # 记忆自压缩：memory_segment 超过此 token 数时,下次提取切到 compact prompt
    MEMORY_SEGMENT_COMPACTION_THRESHOLD = 2000
    # 压缩目标上限（prompt 要求 LLM 压到多少 token 以内）
    MEMORY_SEGMENT_COMPACTED_TARGET = 1500

    def __init__(self,
                 max_tokens: int = 8000,       # LLM 窗口大小
                 reserve_tokens: int = 2000,   # 预留给LLM生成回复的token 大小
                 recent_messages_token: int = 4000,
                 strategy: CompressionStrategy = CompressionStrategy.TASK_AWARE,
                 min_old_token_for_extract: int = 1500,
                 extract_timeout: float = 120.0):

        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens

        self.recent_messages_floor = 4 #至少保留4轮
        self.recent_messages_ceiling  = 30 #最多保留15轮
        self.token_estimate = TokenEstimate()
        self.strategy = strategy
        #提取阈值与超时可配置
        self.recent_token_budget = recent_messages_token
        self.min_old_token_for_extract = min_old_token_for_extract
        self.extract_timeout = extract_timeout
        self._warned_no_llm = False

    @property
    def recent_turns(self) -> int:
        return self.recent_messages_ceiling

    @property
    def recent_messages(self) -> int:
        return self.recent_messages_ceiling

    async def prepare_message(self,
                              system_prompt: str,
                              history: List[Dict[str, str]],
                              llm_client: Optional[LLMClient] = None,
                              task_state: Optional[TaskState] = None) -> List[Dict[str, str]]:

        if self.strategy == CompressionStrategy.TASK_AWARE and task_state:
            messages, protected_head_size = await self._build_aware(
                system_prompt, history, llm_client, task_state)
        elif self.strategy == CompressionStrategy.WINDOW:
            messages = self._build_window(system_prompt, history)
            protected_head_size = 1
        else:
            messages = self._build_preserve_all(system_prompt, history)
            protected_head_size = 1

        # 结构归一化：确保唯一 system 且在最前
        messages = self._ensure_single_system_front(messages)
        # 确保角色是交替进行的 system -> user -> assistant -> user...
        # 注意：若未来切换 OpenAI 原生 tool calling（role="tool"），
        # 本合并会破坏 tool_call_id 对应关系，届时需对 tool 角色加豁免
        messages = self._batch_consecutive_users(messages)
        # 截断必须是最后一道守卫（合并后消息变长，先截后合会失去上限保证）
        messages = self._emergency_truncate(messages, protected_head_size)
        # 协议守卫：截断可能制造孤儿 tool 消息/无人应答的 tool_calls
        messages = self._repair_tool_protocol(messages)

        return messages

    def _build_preserve_all(self, system_prompt: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # 使用 shallow copy 防止对原字典引用的修改
        return [{"role": "system", "content": system_prompt}] + [msg.copy() for msg in history]

    def _build_window(self, system_prompt: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        recent = self._slice_recent_token(history)
        return [{"role": "system", "content": system_prompt}] + [msg.copy() for msg in recent]

    def _slice_recent_token(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        从最新开始累计token,不超过recent_token_budget
        并对齐 User 边界，附带条数硬上下限保护。
        """
        if not history:
            return []

        result:List[Dict[str, str]] = []
        total_token = 0
        budget = self.recent_token_budget
        for msg in reversed(history):
            # 口径与 estimate_message 单条对齐:加 4 tokens 当 role overhead
            msg_token = self.token_estimate.estimate(msg.get("content", "")) + 4

            if ((total_token + msg_token > budget and result) or 
                (len(result) >= self.recent_messages_ceiling)):
                break
            #这里为什么向前插入，原因是我们遍历是从最后开始的
            result.insert(0,msg)
            total_token += msg_token

        start = len(history) - len(result)
        #向前回溯4条消息，一般llm 在 1~3条，4条能满足
        min_start = max(0, start - 4)
        while start > min_start and self._is_orphan(history[start]):
            start -= 1
            result.insert(0, history[start])

        while len(result) < self.recent_messages_floor and start > 0:
            start -= 1
            result.insert(0,history[start])
        
        return result

    @staticmethod
    def _is_orphan(msg: Dict[str, str]) -> bool:
        content = msg.get("content", "")
        if msg.get("role") != "user":
            return True
        if content.startswith("[SYSTEM NOTICE:"):
            return False
        return content.startswith(
            ("Observation", "[Parse Error]", "[Approval", "[SYSTEM", "[System"))

    @staticmethod
    def _batch_consecutive_users(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return messages

        result = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            # 防御：提取 content 时做标准化
            def get_content_safe(m):
                raw = m.get("content")
                if raw is None:
                    return "[空消息]"
                if isinstance(raw, str):
                    return raw
                # 如果是多模态列表（OpenAI 格式），尝试提取文本部分，否则序列化
                if isinstance(raw, list):
                    texts = [part.get("text", "") for part in raw if isinstance(part, dict) and "text" in part]
                    return "\n".join(texts) if texts else "[非文本内容]"
                return str(raw)

            # 检测连续的 user 消息（排除工具注入的虚拟 user）
            if msg["role"] == "user" and "tool_calls" not in msg:
                batch = [msg]
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "user" and "tool_calls" not in messages[j]:
                    batch.append(messages[j])
                    j += 1
                
                if len(batch) > 1:
                    content_parts = [
                        f"[SYSTEM NOTICE: 用户连续发送了 {len(batch)} 条独立消息。"
                        f"请严格按编号顺序理解，注意后续消息可能包含对前述消息的修正、补充或撤销。]"
                    ]
                    for idx, m in enumerate(batch, 1):
                        content_parts.append(f"{idx}. {get_content_safe(m)}")
                    
                    result.append({
                        "role": "user", 
                        "content": "\n".join(content_parts)
                    })
                else:
                    result.append(msg.copy())
                i = j
            else:
                result.append(msg.copy())
                i += 1
        return result

    async def _build_aware(self,
                           system_prompt: str,
                           history: List[Dict[str, str]],
                           llm_client: Optional[LLMClient],
                           task_state: Optional[TaskState]) -> Tuple[List[Dict[str, str]], int]:

        recent = self._slice_recent_token(history)
        cut = len(history) - len(recent)

        #recent 是需要保留的，那么旧消息就是0->(history - recent)
        old = history[:cut] if cut > 0 else []

        if old and task_state:
            if llm_client is None:
                self._warn_no_llm_once()
            # 这里是增量提取，memory_cursor 记录的是上次提取最后条数
            cursor = getattr(task_state, "memory_cursor", 0)

            if cut > cursor:
                pending = history[cursor:cut]
            else:
                pending = []

            if pending:
                pending_tokens = self.token_estimate.estimate_message(pending)

                #绝对 token + 相对 token
                should_extract = (
                    pending_tokens >= self.min_old_token_for_extract
                    or pending_tokens >= self.recent_token_budget * 0.4
                )
                if should_extract:
                    new_memory = await self._extract_task_memory(pending, llm_client, task_state)
                    if new_memory:
                        task_state.memory_segment = new_memory
                        task_state.memory_cursor = cut
                        user_cnt_old = getattr(task_state, "_last_extract_user_count", 0)
                        user_msgs_in_pending = sum(
                            1 for m in pending if m.get("role") == "user"
                        )
                        user_cnt_new = user_cnt_old + user_msgs_in_pending
                        task_state._last_extract_user_count = user_cnt_new
                        _LOG.info("增量提取 step=%d cursor %d→%d (%d条, ~%d tokens, user_cnt %d→%d)",
                                    getattr(task_state, "current_step", 0),
                                    cursor, cut, len(pending), pending_tokens,
                                    user_cnt_old, user_cnt_new)

        # 持久记忆状态 三段 (task_summary / key_facts / memory_segment) 挪出 system,
        # 拼成一条 user 消息 + 一条 assistant 占位,目的是让 SS 永远稳定:
        # 持久记忆状态 变化时只影响 user 那条,SS + plan + assistant 占位
        # 都能持续命中 prefix cache
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        memory_parts: List[str] = []
        if task_state and task_state.task_summary:
            memory_parts.append(f"[Task goal]\n{task_state.task_summary}")
        if task_state and task_state.key_facts:
            facts_text = "\n".join(f"- {f}" for f in task_state.key_facts)
            memory_parts.append(f"[key facts]\n{facts_text}")
        if task_state and task_state.memory_segment:
            memory_parts.append(f"[memory]\n{task_state.memory_segment}")
            
        if getattr(task_state, "_facts_evicted_total", 0) > 0:
            memory_parts.append(
                f"[Facts note] 历史累计已驱逐 {task_state._facts_evicted_total} 条较早事实"
                f"(当前 key_facts 上限 50 条)."
                f"被驱逐的内容仅在 memory_segment 中可能保留,"
                f"如需引用请优先依据 memory_segment."
            )
        if memory_parts:
            # 持久记忆状态走 user 消息 + assistant 占位的目的是让 system + 占位锚定 prefix
            memory_content = (
                "[Persistent task memory — system-managed background "
                "state (task goal, key facts, working memory) auto-injected "
                "for context continuity. This is NOT a user message and the "
                "next real user request will follow. Continue working on the "
                "task as you would for any normal turn.]\n\n"
                + "\n\n".join(memory_parts)
            )
            messages.append({"role": "user", "content": memory_content})
            # assistant 占位:user 跟 recent[0] 都是 user role,
            # 不隔开会被 _merge_consecutive_roles 合并,把 持久记忆状态 拖进 recent 变化区
            messages.append({
                "role": "assistant",
                "content": "Background context noted. Proceeding with the task.",
            })

        messages.extend([msg.copy() for msg in recent])
        return messages, 3 if memory_parts else 1

    def _warn_no_llm_once(self) -> None:
        if not self._warned_no_llm:
            self._warned_no_llm = True
            _LOG.warning("TASK_AWARE 策略需要 llm_client 做记忆提取，当前为 None，"
                         "退化为无记忆窗口（历史消息将被静默丢弃）")

    async def _extract_task_memory(self,
                                   new_old_messages: List[Dict[str, str]],
                                   llm_client: LLMClient,
                                   task_state: TaskState) -> Optional[str]:

        history_text = ""
        # 增量模式：new_old_messages 是"新变旧"的全部（通常 4~几十条），逐条头尾截断控量
        for msg in new_old_messages:
            role = msg.get("role", "UNKNOWN")
            content = msg.get("content", "")

            # 这里防止LLM 提取窗口爆炸，但是对于现在模型上下文窗口1M来说,不需要进行截断进行判断，截断反而在压缩的时候，会造成很多必要消息丢弃
            # if len(content) > 500:
            #     content = content[:250] + "\n...[Middle content omitted]...\n" + content[-250:]

            history_text += f"{role}:{content}\n"

        facts_text = "\n".join(f"- {fact}" for fact in task_state.key_facts) or "None"
        summary_text = task_state.task_summary or "None"

        # prompt 与解析端显式对齐——四项标签、禁代码块、禁解释、无噪音尾行
        prev_memory = task_state.memory_segment or "(无)"
        prev_memory_tokens = self.token_estimate.estimate(prev_memory)
        compaction_mode = prev_memory_tokens > self.MEMORY_SEGMENT_COMPACTION_THRESHOLD

        if compaction_mode:
            size_limit = f"压缩到约 {self.MEMORY_SEGMENT_COMPACTED_TARGET} tokens 以内"
            compaction_directive = (
                f"\n重要:已有记忆已膨胀至约 {prev_memory_tokens} tokens,"
                f"本次输出必须{size_limit}。"
                f"丢弃已彻底完成的步骤细节,仅保留:任务目标、未完成项、关键发现。\n"
            )
        else:
            size_limit = "800字以内"
            compaction_directive = ""

        prompt = textwrap.dedent(f"""\
            请综合"已有记忆"与"本次新增对话"，输出"截至当前的完整任务记忆"({size_limit})。
            - 已有记忆中仍然成立的内容必须保留
            - 新增对话里的关键信息要并入对应分类
            - 已被新对话覆盖或推翻的旧信息可以删除
            - 严格按以下三项输出，每项一个标签；不要输出其他内容，不要使用代码块{compaction_directive}

            <previous_memory>
            {prev_memory}
            </previous_memory>

            <new_history_since_last_extraction>
            {history_text}
            </new_history_since_last_extraction>

            <current_task_summary>
            {summary_text}
            </current_task_summary>

            <current_key_facts>
            {facts_text}
            </current_key_facts>

            任务目标:[一句话概括用户最终想达成什么，如果已有则保持一致]
            已完成:[已完成的关键步骤]
            关键发现:[重要事实、数据、决策，每条用 - 开头]
            """)
        try:
            #提取超时——慢 LLM 不得卡死整个步骤
            raw = await asyncio.wait_for(
                llm_client.chat([{"role": "user", "content": prompt}], tool_schema=None),
                timeout=self.extract_timeout)
        except (asyncio.TimeoutError, TimeoutError):
            _LOG.warning("任务记忆提取超时（>%ss，本次跳过，游标不推进）",
                         self.extract_timeout)
            return None
        except Exception:
            #裸吞异常会让 memory 功能静默失效，必须留痕
            _LOG.warning("任务记忆提取失败（本次跳过，游标不推进）", exc_info=True)
            return None

        #返回值标准化——兼容 str / ChatResponse(有 text 属性) / 其他对象
        result = getattr(raw, "text", raw)
        if not isinstance(result, str):
            result = str(result)
        response_text = result.strip()
        if not response_text:
            _LOG.warning("任务记忆提取返回空响应（本次跳过，游标不推进）")
            return None
        self._parse_and_update_state(response_text, task_state)
        if compaction_mode:
            task_state._compaction_count += 1
            _LOG.info("记忆自压缩触发(prev=%d tokens → target<=%d tokens, 第 %d 次)",
                      prev_memory_tokens, self.MEMORY_SEGMENT_COMPACTED_TARGET,
                      task_state._compaction_count)
        return response_text

    def _parse_and_update_state(self, memory_text: str, task_state: TaskState):
        matched_sections = 0
        goal_match = re.search(r"任务目标[：:]\s*(.+)", memory_text)
        if goal_match:
            matched_sections += 1
            goal = goal_match.group(1).strip()
            # LLM 输出 "None"/"无" 时不得覆盖已有的好 summary
            if goal and goal.lower() not in ("none", "无", "n/a"):
                task_state.task_summary = goal

        # 按 section 解析"关键发现"，兼容 - / • / 数字. 前缀，
        # 不再依赖 LLM 严格使用 "- " 开头
        in_facts = False
        for raw_line in memory_text.split("\n"):
            line = raw_line.strip()
            if re.match(r"^关键发现[：:]", line):
                matched_sections += 1
                in_facts = True
                # 同行内联内容：关键发现: - xxx
                line = re.sub(r"^关键发现[：:]\s*", "", line)
                if not line:
                    continue
            elif re.match(r"^(任务目标|已完成|待处理)\s*[：:]", line):
                in_facts = False
                continue
            if not in_facts:
                continue
            m = re.match(r"^[-•*\d.、]+\s*(.+)$", line)
            fact = (m.group(1) if m else line).strip()
            # 大小写不敏感去重：建索引时统一 fold,保留首次出现的原拼写
            if len(fact) > 2:
                key_folded = fact.casefold()
                existing_keys = {f.casefold() for f in task_state.key_facts}
                if key_folded not in existing_keys:
                    task_state.key_facts.append(fact)
                    # 防止存储爆炸——编码任务的关键事实(API名/路径/版本)很容易超 20
                    if len(task_state.key_facts) > 50:
                        task_state.key_facts.pop(0)
                        task_state._facts_evicted_total += 1
        #四项标签一个都没匹配上 -> 格式不符，留痕（原文仍入 memory_segment）
        if matched_sections == 0:
            _LOG.warning("记忆提取输出格式不符（未匹配到任何标签）：%.80s...", memory_text)

    @staticmethod
    def _merge_consecutive_roles(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return messages

        # 复制第一条消息，避免直接修改原引用
        merged = [messages[0].copy()]

        # 把相邻 role 的角色进行合并
        # 注意：role="tool"（原生 tool calling）绝不合并
        # 每条 tool 消息与 tool_call_id 一一对应，合并会破坏协议；
        # 携带 tool_calls/tool_call_id 键的消息也不参与合并
        # 合并只保留 role/content，协议键会被静默丢弃
        for msg in messages[1:]:
            prev = merged[-1]
            protocol_keys = ("tool_calls", "tool_call_id")
            if (msg["role"] == prev["role"]
                    and msg["role"] not in ("system", "tool")
                    and not any(k in msg or k in prev for k in protocol_keys)):
                # 重新创建一个新 dict 赋值，防止直接修改传进来的外部真实 history 数据
                merged[-1] = {
                    "role": prev["role"],
                    "content": prev["content"] + "\n" + msg["content"]
                }
            else:
                merged.append(msg.copy())

        return merged

    @staticmethod
    def _ensure_single_system_front(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        system_contents = []
        non_system = []
        for message in messages:
            if message["role"] == "system":
                system_contents.append(message["content"])
            else:
                non_system.append(message.copy())

        if not system_contents:
            return non_system

        combined_system = "\n\n".join(system_contents)
        return [{"role": "system", "content": combined_system}] + non_system

    @staticmethod
    def _repair_tool_protocol(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """OpenAI tool 协议：
        - assistant 声明的每个 tool_call id，都必须有对应 role="tool" 响应，
          缺失（被截断）的在其后补占位；
        - 没有归属的孤儿 tool 消息（assistant 被截断）直接删除。
        幂等：消息本就合法时原样返回。
        """
        has_tool = any(m.get("role") == "tool" or m.get("tool_calls") for m in messages)
        if not has_tool:
            return messages

        answered: set = set()
        declared: set = set()
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                declared.update(tc.get("id") for tc in m["tool_calls"] if tc.get("id"))
            elif m.get("role") == "tool" and m.get("tool_call_id"):
                answered.add(m["tool_call_id"])

        cleaned = [m for m in messages
                   if not (m.get("role") == "tool"
                           and m.get("tool_call_id") not in declared)]
        missing = declared - answered
        if not missing:
            return cleaned

        repaired: List[Dict[str, str]] = []
        for m in cleaned:
            repaired.append(m)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc.get("id") in missing:
                        repaired.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "[工具结果因上下文截断丢失]",
                        })
        return repaired

    def _emergency_truncate(self,
                            messages: List[Dict[str, str]],
                            protected_head_size: int = 1) -> List[Dict[str, str]]:
        budget = self.max_tokens - self.reserve_tokens
        if self.token_estimate.estimate_message(messages) <= budget:
            return messages

        # 第二级：整条丢弃最旧的 middle 消息
        keep_tail = 4
        while True:
            non_head_count = max(0, len(messages) - protected_head_size)
            if non_head_count == 0:
                # 没有可丢弃的中间消息，直接进入内容截断
                result = [m.copy() for m in messages]
                break

            tail_n = min(keep_tail, non_head_count)
            head = [m.copy() for m in messages[:protected_head_size]]
            tail = [m.copy() for m in messages[-tail_n:]] if tail_n > 0 else []
            middle = [m.copy() for m in messages[protected_head_size:len(messages)-tail_n]]

            # 基于当前结果计算 token，确保丢弃时计数准确
            # 这里为什么从 最早开始丢弃，原因是，在增量提取的时候，最早的消息已经被摘要提取了
            current_total = self.token_estimate.estimate_message(head + middle + tail)
            while middle and current_total > budget:
                discarded = middle.pop(0)
                current_total -= self.token_estimate.estimate(discarded.get("content", "")) + 4

            result = head + middle + tail
            if self.token_estimate.estimate_message(result) <= budget or keep_tail <= 1:
                break
            keep_tail //= 2

        # 第三级：对 content 做头尾截断
        guard = 0
        while self.token_estimate.estimate_message(result) > budget and guard < 12:
            guard += 1
            candidates = range(1, len(result)) if len(result) > 1 else range(len(result))
            longest_i = max(candidates, key=lambda i: len(result[i].get("content", "")))
            content = result[longest_i].get("content", "")
            if len(content) <= 40:
                break
            quarter = max(20, len(content) // 4)
            new_content = content[:quarter] + "\n...[truncated]...\n" + content[-quarter:]
            if len(new_content) >= len(content):
                break
            result[longest_i] = {**result[longest_i], "content": new_content}

        # 硬收缩（最后手段）
        est_now = self.token_estimate.estimate_message(result)
        if est_now > budget:
            _LOG.warning("紧急截断三级后仍超标（est=%d > budget=%d），执行硬收缩",
                        est_now, budget)
            ratio = max(0.05, budget / est_now * 0.9)
            shrunk = []
            for m in result:
                c = m.get("content", "")
                keep = max(20, int(len(c) * ratio))
                if len(c) > keep:
                    half = max(10, keep // 2)
                    c = c[:half] + "\n...[hard truncated]...\n" + c[-half:]
                shrunk.append({**m, "content": c})
            result = shrunk

            guard = 0
            while self.token_estimate.estimate_message(result) > budget and guard < 50:
                guard += 1
                i = max(range(len(result)),
                        key=lambda j: len(result[j].get("content", "")))
                c = result[i].get("content", "")
                if len(c) <= 20:
                    _LOG.error("全部消息已到硬下限仍超预算（est=%d > budget=%d），"
                            "token 估算可能失真，放行", 
                            self.token_estimate.estimate_message(result), budget)
                    break
                keep = max(20, len(c) // 2)
                half = max(10, keep // 2)
                new_c = c[:half] + "…" + c[-half:]
                if len(new_c) >= len(c):
                    new_c = c[:10] + "…" + c[-10:]
                result[i] = {**result[i], "content": new_c}

        return result