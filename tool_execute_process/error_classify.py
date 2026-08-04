
from enum import Enum, auto
from typing import Optional
import re
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, r'C:\\Users\\zym\\AppData\\Local\\pylibs')
import torch
from transformers import AutoModel,AutoTokenizer
import numpy as np
import logging

_LOG = logging.getLogger("harness.error_classify")


class ToolErrorType(Enum):
    """工具错误类型分类"""
    NETWORK_TIMEOUT = auto()      # 网络超时，可重试
    CONNECTION_ERROR = auto()     # 连接断开，可重试
    RATE_LIMITED = auto()         # 限流,可重试（特殊退避）
    INVALID_PARAMS = auto()       # 参数错误,不可重试，反馈模型修正
    PERMISSION_DENIED = auto()    # 权限不足,不可重试，可能需要人工审批
    BUSINESS_ERROR = auto()       # 业务逻辑错误,不可重试
    TOOL_NOT_FOUND = auto()       # 工具不存在,不可重试
    UNKNOWN = auto()              # 未知错误,有限重试


PATTERNS = {
    ToolErrorType.NETWORK_TIMEOUT: [
        r"timeout",
        r"timed out",
        r"request timeout",
        r"读取超时",
        r"连接超时",
    ],
    ToolErrorType.CONNECTION_ERROR: [
        r"connection",
        r"connect",
        r"reset",
        r"unreachable",
        r"refused",
        r"连接重置",      # 修复4：原缺逗号，三行被拼成一个永不匹配的串
        r"连接被拒绝",
        r"连接失败",
    ],
    ToolErrorType.RATE_LIMITED: [
        r"rate limit",
        r"too many requests",
        r"429",
        r"quota exceeded",
        r"限流",
        r"请求过于频繁",
    ],
    ToolErrorType.PERMISSION_DENIED: [
        r"permission",
        r"forbidden",
        r"unauthorized",
        r"403",
        r"401",
        r"权限",
        r"拒绝访问",
    ],
    ToolErrorType.TOOL_NOT_FOUND: [
        r"not found",
        r"404",
        r"不存在",
        r"未找到",
    ],
}

# 可重试类型
RETRYABLE_TYPES = {
    ToolErrorType.NETWORK_TIMEOUT,
    ToolErrorType.CONNECTION_ERROR,
    ToolErrorType.RATE_LIMITED,
    ToolErrorType.UNKNOWN,
}


class ErrorClassify:
    """正则分类：快、零依赖，覆盖常见错误。默认方案。"""

    @classmethod
    def classify(cls, error_msg: str, error_type: Optional[type] = None):
        error_lower = (error_msg or "").lower()
        for etype, patterns in PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_lower, re.IGNORECASE):
                    return etype
        return ToolErrorType.UNKNOWN

    @classmethod
    def is_retryable(cls, error_type: ToolErrorType) -> bool:
        return error_type in RETRYABLE_TYPES

class LLMErrorClassify:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 similarity_threshold=0.5, fallback_to_re=True):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.fallback_to_re = fallback_to_re
        self._model = None
        self._tokenizer = None
        self._pattern_embd_norm = None
        self._load_failed = False

    def _lazy_load(self) -> bool:
        if self._load_failed:
            return False
        if self._model is not None:
            return True
        try:
            import numpy as np
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._np = np
            self._torch = torch
            self._model = AutoModel.from_pretrained(self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            embds = {}
            for etype, patterns in PATTERNS.items():
                if patterns:
                    embd = self._get_embd(patterns)
                    embds[etype] = np.mean(embd, axis=0)
            #提前归一化，防止每次识别的时候，每次都要计算
            self._pattern_embd_norm = {
                etype: e / (np.linalg.norm(e) + 1e-8) for etype, e in embds.items()
            }
            return True
        except Exception:
            _LOG.warning("LLMErrorClassify 模型加载失败，降级为正则分类", exc_info=True)
            self._load_failed = True
            return False

    def _get_embd(self, text):
        np, torch = self._np, self._torch
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
        attention_mask = inputs["attention_mask"]
        out_last_embd = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(out_last_embd.size()).float()
        sum_embd = torch.sum(out_last_embd * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return (sum_embd / sum_mask).numpy()

    def classify(self, error_msg: str):
        if self.fallback_to_re:
            etype = ErrorClassify.classify(error_msg)
            if etype != ToolErrorType.UNKNOWN:
                return etype
        if not self._lazy_load():
            return ToolErrorType.UNKNOWN
        np = self._np
        embd = self._get_embd([error_msg])[0]
        #1e-8 防止除数为零
        embd_norm = embd / (np.linalg.norm(embd) + 1e-8)
        #这里使用的是余弦相似
        #similarity_threshold 置信度
        for etype, pattern_embd in self._pattern_embd_norm.items():
            if float(np.dot(embd_norm, pattern_embd)) > self.similarity_threshold:
                return etype
        return ToolErrorType.UNKNOWN

    @staticmethod
    def is_retryable(error_type: ToolErrorType) -> bool:
        return error_type in RETRYABLE_TYPES
    