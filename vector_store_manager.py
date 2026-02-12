import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.vectorstores import VectorStore
from config import Config

# 创建，保存，加载
class VectorStoreManager:    

    # 初始化 store_type: 向量存储类型 (faiss/chroma) persist_directory: 持久化目录 embedding_model: 嵌入模型名称
    def __init__(self, 
                 store_type: str = None,
                 persist_directory: str = None,
                 embedding_model: str = None):
        
        self.store_type = store_type or Config.VECTOR_STORE_TYPE
        self.persist_directory = persist_directory or Config.VECTOR_STORE_PATH
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        
        # 初始化Embeddings - 支持本地和远程
        if Config.USE_LOCAL_EMBEDDING:
            print(f"使用本地 Embedding 模型: {Config.LOCAL_EMBEDDING_MODEL}")
            print(f"使用设备: {Config.EMBEDDING_DEVICE}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.LOCAL_EMBEDDING_MODEL,
                model_kwargs={'device': Config.EMBEDDING_DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            print(f"使用远程 Embedding 模型: {self.embedding_model}")
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=Config.OPENAI_API_KEY,
                openai_api_base=Config.OPENAI_API_BASE
            )
        
        self.vector_store: Optional[VectorStore] = None
    
    # 创建向量存储 Args:documents: 文档列表 Returns:向量存储对象
    def create_vector_store(self, documents: List[Document]) -> VectorStore:

        print(f"正在创建 {self.store_type} 向量存储...")
        
        if self.store_type.lower() == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        elif self.store_type.lower() == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            raise ValueError(f"不支持的向量存储类型: {self.store_type}")
        
        print(f"向量存储创建完成，包含 {len(documents)} 个文档")
        return self.vector_store
    
    # 保存向量存储到磁盘
    def save_vector_store(self):
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        if self.store_type.lower() == "faiss":
            save_path = os.path.join(self.persist_directory, "faiss_index")
            self.vector_store.save_local(save_path)
            print(f"FAISS向量存储已保存到: {save_path}")
        elif self.store_type.lower() == "chroma":
            # Chroma会自动持久化
            print(f"Chroma向量存储已保存到: {self.persist_directory}")
    
    # 从磁盘加载向量存储 Returns:向量存储对象，如果不存在则返回None
    def load_vector_store(self) -> Optional[VectorStore]:
        if self.store_type.lower() == "faiss":
            save_path = os.path.join(self.persist_directory, "faiss_index")
            if os.path.exists(save_path):
                print(f"正在加载FAISS向量存储从: {save_path}")
                self.vector_store = FAISS.load_local(
                    save_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("FAISS向量存储加载完成")
                return self.vector_store
        elif self.store_type.lower() == "chroma":
            if os.path.exists(self.persist_directory):
                print(f"正在加载Chroma向量存储从: {self.persist_directory}")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print("Chroma向量存储加载完成")
                return self.vector_store
        
        print("未找到已保存的向量存储")
        return None
    
    # 相似度搜索 Args:query: 查询文本 k: 返回文档数量 Returns:相关文档列表
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        
        k = k or Config.RETRIEVAL_K
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    # 获取检索器 Args:k: 返回文档数量 Returns:检索器对象
    def get_retriever(self, k: int = None):
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        
        k = k or Config.RETRIEVAL_K
        return self.vector_store.as_retriever(search_kwargs={"k": k})
