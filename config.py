import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:

    # OpenAI配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Embedding模型配置
    USE_LOCAL_EMBEDDING = os.getenv("USE_LOCAL_EMBEDDING", "true").lower() == "true"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # cpu 或 cuda
    
    # 文本分割配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # 检索配置
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))  # 检索文档数量
    
    # 向量存储配置
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")  # faiss 或 chroma
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    
    # 知识库文件路径
    KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./car_corpus.pdf")
    
    # 温度参数
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    

    @classmethod # 验证必要配置是否存在
    def validate(cls):
        # 使用远程 Embedding 但没有 API Key
        if not cls.USE_LOCAL_EMBEDDING and not cls.OPENAI_API_KEY:
            raise ValueError("使用远程 Embedding 但没有设置 OPENAI_API_KEY 环境变量")
        # 使用远程 LLM 但没有 API Key
        if not cls.OPENAI_API_KEY:
            print("使用远程 LLM 但没有设置 OPENAI_API_KEY 环境变量")
        return True
