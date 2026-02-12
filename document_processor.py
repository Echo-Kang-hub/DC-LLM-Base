"""
文档处理模块 - 负责加载和分割PDF文档
"""
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import Config


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        初始化文档处理器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,  # 添加起始索引
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        加载PDF文件
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            文档列表
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档
        
        Args:
            documents: 原始文档列表
            
        Returns:
            分割后的文档列表
        """
        splits = self.text_splitter.split_documents(documents)
        return splits
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        处理PDF文件：加载并分割
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            分割后的文档列表
        """
        print(f"正在加载PDF文件: {pdf_path}")
        documents = self.load_pdf(pdf_path)
        print(f"加载了 {len(documents)} 个页面")
        
        print("正在分割文档...")
        splits = self.split_documents(documents)
        print(f"分割为 {len(splits)} 个文本块")
        
        return splits
