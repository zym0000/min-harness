
from typing import Dict,List
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#这个sys.path.insert是因为我本地python 库因为window 长度显示，所以把库安装指定位置
import sys
sys.path.insert(0, r'C:\\Users\\zym\\AppData\\Local\\pylibs')
import torch
from transformers import AutoTokenizer, AutoModel

class InputGateway:
    '''
        输入网关,负责对输入数据进行处理
    '''
    DEFAULT_INTENT_EXAMPLES: Dict[str, List[str]] = {
        "file_read": [
            "读取 main.py 的内容",
            "打开 config.yaml 看看",
            "显示这个文件的代码",
            "查看 utils.py 第 10 到 50 行",
            "show me the content of app.py",
            "read the file",
            "cat README.md",
            "这个文件里写了什么",
            "看看 package.json",
        ],
        "file_write": [
            "创建一个新文件 hello.py",
            "写入内容到 output.txt",
            "新建一个 utils 模块",
            "生成一个配置文件",
            "create a new file called test.py",
            "write this code to main.py",
            "保存这段代码",
            "帮我写一个脚本",
            "创建 requirements.txt",
        ],
        "file_patch": [
            "修改 main.py 里的函数名",
            "把第 20 行改成 print",
            "重构这个类",
            "替换所有的 tab 为空格",
            "edit the function to add error handling",
            "patch this bug",
            "更新 API 接口的返回值",
            "给这个函数加个参数",
            "删除第 5 到 10 行",
            "优化这段代码的性能",
        ],
        "search": [
            "搜索哪里用了 UserService",
            "查找所有包含 TODO 的文件",
            "grep 一下 handle_error",
            "哪些文件 import 了 numpy",
            "find all usages of this function",
            "search for the class definition",
            "定位 process_data 函数在哪",
            "项目里有没有用到 redis",
            "搜索所有的 print 语句",
        ],
        "navigate": [
            "列出项目目录结构",
            "看看有哪些文件",
            "显示 src 目录下的内容",
            "ls -la",
            "list all files in the project",
            "show directory tree",
            "这个项目有什么文件",
            "查看文件夹结构",
        ],
        "shell": [
            "运行 python main.py",
            "执行 pip install flask",
            "跑一下 npm run build",
            "运行 docker compose up",
            "execute this command",
            "run the server",
            "安装 requests 库",
            "启动开发服务器",
            "kill 掉 8080 端口的进程",
        ],
        "test": [
            "运行测试",
            "跑一下 pytest",
            "执行单元测试",
            "run the tests",
            "npm test",
            "验证一下改动有没有问题",
            "检查代码是否正确",
            "go test ./...",
            "测试覆盖率是多少",
        ],
        "git": [
            "提交代码",
            "git commit",
            "创建一个新分支",
            "push 到远程",
            "查看 git log",
            "git diff 看看改了什么",
            "回滚上一次提交",
            "merge 这个分支",
            "查看当前的 git status",
        ],
        "project": [
            "初始化一个 FastAPI 项目",
            "搭建项目脚手架",
            "创建一个 React 应用",
            "setup a new project",
            "scaffold a Django app",
            "帮我实现一个完整的功能",
            "开发一个 REST API",
            "写一个爬虫",
            "实现用户登录功能",
        ],
    }

    # 内置默认关键词规则
    DEFAULT_KEYWORD_RULES: Dict[str, List[str]] = {
        "file_read": [
            "读取", "查看", "打开", "显示", "内容",
            "read", "cat", "show", "display", "open",
        ],
        "file_write": [
            "创建", "新建", "写入", "生成", "保存", "写一个",
            "create", "write", "new file", "generate", "save",
        ],
        "file_patch": [
            "修改", "编辑", "替换", "重构", "更新", "删除",
            "改成", "加个", "优化", "修复", "fix", "patch",
            "edit", "refactor", "replace", "update", "modify",
            "rename", "remove", "delete", "change",
        ],
        "search": [
            "搜索", "查找", "定位", "哪里", "grep", "find",
            "search", "locate", "where", "哪些", "有没有",
        ],
        "navigate": [
            "目录", "文件列表", "结构", "ls", "tree",
            "list", "directory", "folder", "有哪些文件",
        ],
        "shell": [
            "运行", "执行", "安装", "启动", "跑一下",
            "run", "exec", "install", "pip", "npm",
            "docker", "command", "命令", "kill",
        ],
        "test": [
            "测试", "验证", "检查", "pytest", "test",
            "unittest", "coverage", "assert",
        ],
        "git": [
            "git", "commit", "push", "pull", "branch",
            "merge", "提交", "分支", "回滚", "revert",
        ],
        "project": [
            "项目", "初始化", "搭建", "开发", "实现",
            "scaffold", "setup", "init", "build",
            "功能", "模块", "系统", "应用",
        ],
    }

    # 无意图时的默认标签集合
    DEFAULT_TAGS = ["general", "file_read", "navigate", "search"]

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 similarity_threshold = 0.8,
                 fallback_to_keywords = True,
                 max_tags = 5):
        # 加载 tokenizer 和模型
        # 词嵌入模型，用来把用户输入的语句和模板进行相似度匹配
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.similarity_threshold = similarity_threshold #similarity_threshold 相似度,只有超过这个才会被采纳
        self.fallback_to_keywords = fallback_to_keywords #是否启动兜底方案。
        self.max_tags = max_tags
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model.to(self.device)
        # 计算例子意图的向量 = 所有例子平均词向量
        self.intent_embeddings: Dict[str, np.ndarray] = {}
        for tag, examples in self.DEFAULT_INTENT_EXAMPLES.items():
            if examples:
                emb = self._get_embedding(examples)
                self.intent_embeddings[tag] = emb.mean(axis=0)
        
        # 归一化向量，避免后续计算余弦相似度时重复计算模长
        self.intent_embeddings_norm = {
            tag: emb / (np.linalg.norm(emb) + 1e-8) 
            for tag, emb in self.intent_embeddings.items()
        }

    def _get_embedding(self, text: str):
        """平均池化"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        #inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 取最后一层隐藏状态，在序列维度上做平均
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        #mask 张量对齐词张量维度，不对齐的话，计算会报错(B,T)->(B,T,1)->(B,T,H)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #token_embeddings * input_mask_expanded 逐元素相乘
        #这里dim = 1 就是第一列逐列相加，每一列都是一个特征，现在有seq长度的词，所有词合成一句话，所以这句话在这一维特征就是所有词相加
        #所以维度相加，就是这句话的特征值(B,H)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        #平均，先算每个句子的词的数量，在把求和的值逐个/总个数
        #（B,T,h) sum(1)做的就是把所以词的每一维都+1 所以没有填充的就是1 ，填充部位就是0，所有1的地方相加就是总词数
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

        if not sorted_intents:
            if self.fallback_to_keywords:
                return self._classify_intent_keyword(text)
            return []
        
        return [tag for tag,_ in sorted_intents]
    
    #关键词匹配，兜底方案
    def _classify_intent_keyword(self,text:str):
        text_lower = text.lower()
        hit_counts: Dict[str, int] = {}

        for tag, keywords in self.DEFAULT_KEYWORD_RULES.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                hit_counts[tag] = count

        if not hit_counts:
            return []

        sorted_tags = sorted(hit_counts.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in sorted_tags[:self.max_tags]]

if __name__ == "__main__":
    gateway = InputGateway()

    test_cases = [
        "帮我读取 main.py 的内容",
        "创建一个新文件 utils.py",
        "修改 handle_request 函数，加上异常处理",
        "搜索项目里哪里用了 Redis",
        "列出 src 目录下的所有文件",
        "运行 pytest 看看测试通不通过",
        "git commit 提交一下",
        "帮我实现一个用户注册的 API",
        "pip install fastapi",
        "把第 20 行的 print 改成 logging.info",
        "",
        "你好",
        "今天天气怎么样",
    ]

    for text in test_cases:
        tags = gateway.process(text)
        print(f"  {text:<40} → {tags}")