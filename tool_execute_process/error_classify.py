
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

PATTERNS ={
    ToolErrorType.NETWORK_TIMEOUT:[
        r"timeout",
        r"timed out",
        r"request timeout",
        r"读取超时",
        r"连接超时",
    ],
    ToolErrorType.CONNECTION_ERROR:[
        r"connection",
        r"connect",
        r"reset",
        r"unreachable",
        r"refused",
        r"连接重置"
        r"连接被拒绝"
        r"连接失败"
    ],
    ToolErrorType.RATE_LIMITED:[
        r"rate limit",
        r"too many requests",
        r"429",
        r"quota exceeded",
        r"限流",
        r"请求过于频繁"
    ],
    ToolErrorType.PERMISSION_DENIED:[
        r"permission",
        r"forbidden",
        r"unauthorized",
        r"403",
        r"401",
        r"权限",
        r"拒绝访问"
    ],
    ToolErrorType.TOOL_NOT_FOUND: [
        r"not found",
        r"404",
        r"不存在",
        r"未找到",
        ],
}

#可重试类型
RETRYABLE_TYPES = {
    ToolErrorType.NETWORK_TIMEOUT,
    ToolErrorType.CONNECTION_ERROR,
    ToolErrorType.RATE_LIMITED,
    ToolErrorType.UNKNOWN,
}

#如果为了追求速度和效率，不考虑很高的准确度，
#直接使用正则进行错误分类 ErrorClassify
#如果考虑准确度，可以选择LLMErrorClassify 进行错误分类.
class ErrorClassify:
    @classmethod
    def classify(cls,error_msg:str, error_type:Optional[type] = None):

        error_lower = error_msg.lower()

        for error_type, patterns in PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_lower, re.IGNORECASE):
                    return error_type

        return ToolErrorType.UNKNOWN
    
    @classmethod
    def is_retryable(cls, error_type: ToolErrorType) -> bool:
        return error_type in RETRYABLE_TYPES
    
class LLMErrorClassify:
    def __init__(self, model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 similarity_threshold = 0.5,
                 fallback_to_re = True):
        
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.similarity_threshold = similarity_threshold
        self.fallback_to_re = fallback_to_re

        classfiy_re_embd = {}

        for error_type, patterns in PATTERNS.items():
            if patterns:
                print(f"patterns : {patterns}")
                embd = self._get_embd(patterns)
                classfiy_re_embd[error_type] = np.mean(axis = 0)

        #做归一化，避免计算的时候再次计算
        self.classify_re_embd_norm = {
            error_type: embd /(np.linalg.norm(embd) + 1e-8)
            for error_type,embd in classfiy_re_embd.items()
        }

    def _get_embd(self,text:str):
        inputs = self.tokenizer(text,return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        attention_mask = inputs["attention_mask"] #shape (B,T)
        out_last_embd = outputs.last_hidden_state #shape (B,T,D)
        input_attention_mask = attention_mask.unsqueeze(-1).expand(out_last_embd.size()).float() #(B,T,D)
        #mena pool
        sum_embd =  sum(out_last_embd * input_attention_mask,dim=1)
        sum_mask = torch.clamp(sum(input_attention_mask,dim=1),min= 1e-9)

        embedding = sum_embd / sum_mask

        return embedding.numpy()
    
    @classmethod
    def classify(self, error_msg):
        if PATTERNS is None:
            return ToolErrorType.UNKNOWN
        
        if self.fallback_to_re:
            error_type = ErrorClassify.classify(error_msg)
            if error_type  == ToolErrorType.UNKNOWN:
                error_type = self._classify_process(error_msg)

            return error_type
            
        return self._classify_process(error_msg)
            
    def _classify_process(self,error_msg):
        error_embd = self._get_embd(error_msg)[0]
        error_embd_norl = error_embd / np.linalg.norm(error_embd)

        for error_type, embd in self.classify_re_embd_norm.items():
            sim = np.dot(error_embd_norl,embd)
            if sim > self.similarity_threshold:
                return error_type
            
        return ToolErrorType.UNKNOWN
    
    @classmethod
    def is_retryable(cls, error_type: ToolErrorType) -> bool:
        return error_type in RETRYABLE_TYPES

                
        



        
            
            



    


