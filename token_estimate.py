from typing import Dict
import re

#以deepseek 为例
#1 个英文字符 ≈ 0.3 个 token。
#1 个中文字符 ≈ 0.6 个 token。
#所有估算token的话，1个token ≈ 1.6中文，4个英文

#这里只是快速进行估算，实际要准确的话，需要下载对应模型tokenizer，调用encode
#返回实际的token
class TokenEstimate:

    @classmethod
    def estimate(cls,text:str):
        if not text:
            return 0
        
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - cn_chars
        return cn_chars/1.6 + other_chars / 4
        
    @classmethod
    def estimate_message(cls, messages: list[Dict[str,str]]):
        total = 0
        for msg in messages:
            total += cls.estimate(msg.get("content",""))
            #这里+4的原因是因为真实场景中，传给LLM的字符会被变成特定格式
            #<|im_start|>user
            #输入的内容
            #<|im_end|>
            #<|assistant_start|> assistant
            #LLm回答
            #<|assistant_end|>
            total += 4
        return total

