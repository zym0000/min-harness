from token_estimate import TokenEstimate
from typing import Dict, Optional, List
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

    # 增量提取触发阈值的默认值
    MIN_NEW_MESSAGES_FOR_EXTRACT = 6

    def __init__(self,
                 max_tokens: int = 8000,       # LLM 窗口大小
                 reserve_tokens: int = 2000,   # 预留给LLM生成回复的token 大小
                 keep_recent_turns: int = 3,   # 保留最近的轮数
                 strategy: CompressionStrategy = CompressionStrategy.TASK_AWARE,
                 min_new_messages_for_extract: int = MIN_NEW_MESSAGES_FOR_EXTRACT,
                 extract_timeout: float = 120.0):

        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        # 注意单位：内部一律按"消息条数"计（1 轮 = user+assistant 2 条）
        self.recent_messages = keep_recent_turns * 2
        self.token_estimate = TokenEstimate()
        self.strategy = strategy
        #提取阈值与超时可配置
        self.min_new_messages_for_extract = min_new_messages_for_extract
        self.extract_timeout = extract_timeout
        self._warned_no_llm = False

    # 兼容旧引用：self.recent_turns 历史上存的是"条数"，保留别名避免外部依赖破裂
    @property
    def recent_turns(self) -> int:
        return self.recent_messages

    async def prepare_message(self,
                              system_prompt: str,
                              history: List[Dict[str, str]],
                              llm_client: Optional[LLMClient] = None,
                              task_state: Optional[TaskState] = None) -> List[Dict[str, str]]:

        if self.strategy == CompressionStrategy.TASK_AWARE and task_state:
            messages = await self._build_aware(system_prompt, history, llm_client, task_state)
        elif self.strategy == CompressionStrategy.WINDOW:
            messages = self._build_window(system_prompt, history)
        else:
            messages = self._build_preserve_all(system_prompt, history)

        # 结构归一化：确保唯一 system 且在最前
        messages = self._ensure_single_system_front(messages)
        # 确保角色是交替进行的 system -> user -> assistant -> user...
        # 注意：若未来切换 OpenAI 原生 tool calling（role="tool"），
        # 本合并会破坏 tool_call_id 对应关系，届时需对 tool 角色加豁免
        messages = self._merge_consecutive_roles(messages)
        # 截断必须是最后一道守卫（合并后消息变长，先截后合会失去上限保证）
        messages = self._emergency_truncate(messages)
        # 协议守卫：截断可能制造孤儿 tool 消息/无人应答的 tool_calls
        messages = self._repair_tool_protocol(messages)

        return messages

    def _build_preserve_all(self, system_prompt: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # 使用 shallow copy 防止对原字典引用的修改
        return [{"role": "system", "content": system_prompt}] + [msg.copy() for msg in history]

    def _build_window(self, system_prompt: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        recent = self._slice_recent_aligned(history)
        return [{"role": "system", "content": system_prompt}] + [msg.copy() for msg in recent]

    def _slice_recent_aligned(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        取最近 recent_messages 条，并对齐"轮边界"：
        若切片开头是孤儿消息（assistant 的 Action / 半截 Observation），
        向前扩展（最多多带 4 条）直到一个真实用户输入——
        宁可窗口略大，也不给模型看没有前因的半截上下文。
        """
        if len(history) <= self.recent_messages:
            return list(history)
        start = len(history) - self.recent_messages
        min_start = max(0, start - 4)
        while start > min_start and self._is_orphan(history[start]):
            start -= 1
        return history[start:]

    @staticmethod
    def _is_orphan(msg: Dict[str, str]) -> bool:
        content = msg.get("content", "")
        if msg.get("role") != "user":
            return True
        return content.startswith(
            ("Observation", "[Parse Error]", "[Approval", "[SYSTEM", "[System"))

    async def _build_aware(self,
                           system_prompt: str,
                           history: List[Dict[str, str]],
                           llm_client: Optional[LLMClient],
                           task_state: Optional[TaskState]) -> List[Dict[str, str]]:

        recent = self._slice_recent_aligned(history)
        #recent 是需要保留的，那么旧消息就是0->(history - recent)
        old = history[:len(history) - len(recent)] if len(history) > len(recent) else []

        if old and task_state:
            if llm_client is None:
                self._warn_no_llm_once()
            else:
                # 这里是增量提取，memory_cursor 记录的是上次提取最后条数
                cursor = getattr(task_state, "memory_cursor", 0)
                cut = len(history) - len(recent)
                new_old = history[cursor:cut]
                # 连续失败指数退避（×2^failures，上限 8 倍），
                # 避免 LLM 持续故障时每一步都白调一次
                failures = getattr(task_state, "_extract_failures", 0)
                #这里不是每次都提取，只要满足最小的提取条数才会触发
                threshold = self.min_new_messages_for_extract * (2 ** min(failures, 3))
                if len(new_old) >= threshold:
                    #更新提取信息
                    new_memory = await self._extract_task_memory(new_old, llm_client, task_state)
                    if new_memory:
                        task_state.memory_segment = new_memory
                        task_state.memory_cursor = cut   # 提取成功才推进，失败下步重试
                        task_state._extract_failures = 0
                        _LOG.info("增量记忆提取完成：游标 %d->%d，%d 条消息",
                                  cursor, cut, len(new_old))
                    else:
                        task_state._extract_failures = failures + 1

        system_parts = [system_prompt]
        if task_state and task_state.task_summary:
            system_parts.append(f"[Task goal]: {task_state.task_summary}")

        if task_state and task_state.key_facts:
            facts_text = "\n".join(f"- {f}" for f in task_state.key_facts)
            system_parts.append(f"[key facts]\n{facts_text}")

        #memory 折叠进 system
        if task_state and task_state.memory_segment:
            system_parts.append(
                f"[memory · 累积记忆截至第 {task_state.memory_cursor} 条历史]\n"
                f"{task_state.memory_segment}")

        system = "\n\n".join(system_parts)
        messages = [{"role": "system", "content": system}]
        messages.extend([msg.copy() for msg in recent])
        return messages

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

            # 采用 头...尾 的提取方式，避免中间截断导致关键语义完全丢失
            if len(content) > 500:
                content = content[:250] + "\n...[Middle content omitted]...\n" + content[-250:]

            history_text += f"{role}:{content}\n"

        facts_text = "\n".join(f"- {fact}" for fact in task_state.key_facts) or "None"
        summary_text = task_state.task_summary or "None"

        # prompt 与解析端显式对齐——四项标签、禁代码块、禁解释、无噪音尾行
        prev_memory = task_state.memory_segment or "(无)"
        prompt = textwrap.dedent(f"""\
            请综合"已有记忆"与"本次新增对话"，输出"截至当前的完整任务记忆"（800字以内）。
            - 已有记忆中仍然成立的内容必须保留
            - 新增对话里的关键信息要并入对应分类
            - 已被新对话覆盖或推翻的旧信息可以删除
            - 严格按以下四项输出，每项一个标签；不要输出其他内容，不要使用代码块

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
            待处理:[还需要做的]
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
            if len(fact) > 2 and fact not in task_state.key_facts:
                task_state.key_facts.append(fact)
                # 防止存储爆炸
                if len(task_state.key_facts) > 20:
                    task_state.key_facts.pop(0)
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

    def _emergency_truncate(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        三级防线：
        1 不超标直接返回（不再有 len<=5 豁免 5 条大消息照样能打爆窗口）；
        2 整条丢弃最旧的 middle 并插入占位标记；若保尾 4 条本身就超标，
           保尾数从 4 -> 2 -> 1 递减（至少保 1 条，丢光尾部不如截断它）；
        3 仍超标 -> 对 content 做头尾截断（保 role 结构，system 最后才动），
           下限 40 字符，无进展即停，保证收敛不死循环。
        """

        budget = self.max_tokens - self.reserve_tokens
        if self.token_estimate.estimate_message(messages) <= budget:
            return messages

        #第二级：整条丢弃
        # 这里丢弃 先丢弃最早的信息，如果可以丢弃的信息都丢弃了，还是超了，那么从剩余末尾开始丢弃
        #末尾的丢弃策略是4-2->1 
        keep_tail = 4
        while True:
            tail_n = max(1, min(keep_tail, len(messages) - 1))
            head = [messages[0].copy()]
            tail = [m.copy() for m in messages[len(messages) - tail_n:]]
            middle = [m.copy() for m in messages[1:len(messages) - tail_n]]

            current_total = self.token_estimate.estimate_message(messages)
            dropped = 0
            while middle and current_total > budget:
                discarded = middle.pop(0)
                dropped += 1
                current_total -= self.token_estimate.estimate(discarded.get("content", "")) + 4

            result = head + middle + tail
            if dropped:
                # 告诉模型有上下文被丢弃，不要静默消失
                result.insert(1, {
                    "role": "user",
                    "content": f"[{dropped} 条早期上下文因长度限制已丢弃]"
                })

            if self.token_estimate.estimate_message(result) <= budget or keep_tail <= 1:
                break
            keep_tail //= 2

        #如果丢弃消息还是超了,截断 content
        #这里要说明下，如果走到这里，那么整个记忆系统里面 system +  最终一条信息
        guard = 0
        while self.token_estimate.estimate_message(result) > budget and guard < 12:
            guard += 1
            # 优先截非 system 的最长消息；没有其它消息时才动 system
            candidates = range(1, len(result)) if len(result) > 1 else range(len(result))
            longest_i = max(candidates, key=lambda i: len(result[i].get("content", "")))
            content = result[longest_i].get("content", "")
            #小于40 不截取
            if len(content) <= 40:
                break
            #这里是截断一半，前面留1/4，后面留1/4
            quarter = max(20, len(content) // 4)
            new_content = content[:quarter] + "\n...[truncated]...\n" + content[-quarter:]
            if len(new_content) >= len(content):
                break
            result[longest_i] = {**result[longest_i], "content": new_content}

        # 硬收缩
        est_now = self.token_estimate.estimate_message(result)
        if est_now > budget:
            _LOG.warning("紧急截断三级后仍超标（est=%d > budget=%d），执行硬收缩",
                         est_now, budget)
            #这里是等比例收缩，*0.9是为了保留10% 估算误差
            ratio = max(0.05, budget / est_now * 0.9)
            shrunk = []
            #对所有消息进行比例压缩
            for m in result:
                c = m.get("content", "")
                keep = max(20, int(len(c) * ratio))
                if len(c) > keep:
                    half = max(10, keep // 2)
                    c = c[:half] + "\n...[hard truncated]...\n" + c[-half:]
                shrunk.append({**m, "content": c})
            result = shrunk

            #如果还是超了，那么找打最长消息，进行对半砍
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

                #对半砍实现
                keep = max(20, len(c) // 2)
                half = max(10, keep // 2)
                new_c = c[:half] + "…" + c[-half:]
                if len(new_c) >= len(c):            # 无进展兜底：直接砍到下限
                    new_c = c[:10] + "…" + c[-10:]
                result[i] = {**result[i], "content": new_c}

        return result