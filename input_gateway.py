
from typing import Dict
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, r'C:\\Users\\zym\\AppData\\Local\\pylibs')
import torch
from transformers import AutoTokenizer, AutoModel

class InputGateway:
    '''
        输入网关,负责对输入数据进行处理
    '''
    INTENT_EXAMPLES ={
        "weather": [
            "今天天气怎么样",
            "北京会下雨吗",
            "现在气温多少度",
            "需要带伞吗",
        ],
        "calculate": [
            "帮我算一下 15*23",
            "1+1等于几",
            "计算这个表达式",
            "(100+50)/3 的结果",
        ],
        "file": [
            "读取 report.txt",
            "打开这个文件看看",
            "显示文件内容",
            "查看文档",
        ],
        "database": [
            "查询用户表",
            "数据库里搜索订单",
            "SELECT * FROM users",
            "帮我查一下数据库",
        ],
        "email": [
            "发送邮件给老板",
            "给张三发一封邮件",
            "写邮件通知团队",
            "email 主题是会议通知",
        ],
    }

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 similarity_threshold = 0.5,
                 fallback_to_keywords = True):
        # 加载 tokenizer 和模型
        #词嵌入模型，用来把用户输入的语句和模板进行相似度匹配
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.similarity_threshold = similarity_threshold #similarity_threshold 相似度,只有超过这个才会被采纳
        self.fallback_to_keywords = fallback_to_keywords #是否启动兜底方案。
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model.to(self.device)
        # 计算例子意图的向量 = 所有例子平均词向量
        self.intent_embeddings: Dict[str, np.ndarray] = {}
        for tag, examples in self.INTENT_EXAMPLES.items():
            if examples:
                emb = self._get_embedding(examples)
                self.intent_embeddings[tag] = emb.mean(axis=0)
        
        # 归一化向量，避免后续计算余弦相似度时重复计算模长
        self.intent_embeddings_norm = {
            tag: emb / (np.linalg.norm(emb) + 1e-8) 
            for tag, emb in self.intent_embeddings.items()
        }

    def _get_embedding(self, text: str):
        """对句子进行平均池化"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        #inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 取最后一层隐藏状态，在序列维度上做平均
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #token_embeddings * input_mask_expanded 逐元素相乘
        #这里dim = 1 逐列相加，每一列都是一个特征，现在有seq长度的词，所有词合成一句话，所以这句话在这一维特征就是所有词相加
        #所以维度相加，就是这句话的特征值
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        #平均，先算每个句子的词的数量，在把求和的值逐个/总个数
        #（B,T,h) sum(1)做的就是把所以词的每一维都+1 所以没有填充的就是1 ，填充部位就是0，所有1的地方相加就是总词数
        #这里要明白input_mask_expanded 是经过广播扩展的形状(B,T)->(B,T,H) 为了对齐token_embeddings 形状
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        #这里做平均
        embedding = sum_embeddings / sum_mask
        #最后就是平均词化 embedding
        return embedding.numpy()

    def process(self, text: str):
        return self._classify_intent(text)

    def _classify_intent(self,text:str):
        if not self.intent_embeddings:
            return []
        
        query_embd = self._get_embedding([text])[0] 
        query_emb_norm = query_embd / (np.linalg.norm(query_embd) + 1e-8)
        scores = {}

        for tag,intent_emb in self.intent_embeddings_norm.items():
            #余弦相似度 A*B/||A||*||B|| 这里* 是点积
            #本质就是相似的token 在向量上，他们方向会很相近，
            #向量本质代表空间的方向和长度，余弦相似度 代表是方向的一致性
            sim = np.dot(query_emb_norm,intent_emb)
            if sim > self.similarity_threshold:
                scores[tag] = float(sim)

        sorted_intents = sorted(scores.items(),key=lambda x:x[1],reverse=True)

        if not sorted_intents :
            if self.fallback_to_keywords:
                return self._classify_intent_keyword(text)
            return []
        
        return [tag for tag,_ in sorted_intents]
    
    #关键词兜底方案
    def _classify_intent_keyword(self,text:str):
        text_lower = text.lower()
        tags = []
        if any(kw in text_lower for kw in ["天气", "weather", "气温", "下雨"]):
            tags.append("weather")
        if any(kw in text_lower for kw in ["计算", "等于", "arithmetic"]):
            tags.append("calculate")
        if any(kw in text_lower for kw in ["文件", "读取", "file", "read"]):
            tags.append("file")
        if any(kw in text_lower for kw in ["数据库", "查询", "sql", "search"]):
            tags.append("database")
        if any(kw in text_lower for kw in ["邮件", "email", "发送"]):
            tags.append("email")
        return tags


if __name__ == "__main__":
    gateway = InputGateway()
    print(gateway.process("今天北京天气如何？"))
    print(gateway.process("帮我算一下 123*456"))