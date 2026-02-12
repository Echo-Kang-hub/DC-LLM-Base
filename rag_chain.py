"""
RAG链模块 - 实现检索增强生成
"""
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import Config
from vector_store_manager import VectorStoreManager


class RAGChain:
    """RAG问答链"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        初始化RAG链
        
        Args:
            vector_store_manager: 向量存储管理器
        """
        self.vector_store_manager = vector_store_manager
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE
        )
        
        # 创建提示模板
        self.system_prompt = """你是一个专业的汽车知识助手。你的任务是根据提供的上下文信息回答用户的问题。

回答要求：
1. 仅根据提供的上下文信息回答问题
2. 如果上下文中没有相关信息，请明确告知用户"根据现有资料无法回答这个问题"
3. 回答要准确、简洁、专业
4. 可以引用上下文中的具体内容
5. 如果问题与汽车知识无关，请礼貌地告知用户你只能回答汽车相关的问题

上下文信息：
{context}
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}")
        ])
        
        # 创建检索器
        self.retriever = self.vector_store_manager.get_retriever()
        
        # 对话历史
        self.chat_history: List[Any] = []
    
    def _format_docs(self, docs):
        """格式化文档"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def invoke(self, question: str, use_history: bool = True) -> Dict[str, Any]:
        """
        调用RAG链回答问题
        
        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            
        Returns:
            包含答案和上下文的字典
        """
        # 检索相关文档
        retrieved_docs = self.retriever.invoke(question)
        context = self._format_docs(retrieved_docs)
        
        # 准备消息
        messages = [
            ("system", self.system_prompt.format(context=context)),
        ]
        
        # 添加历史
        if use_history and self.chat_history:
            messages.append(MessagesPlaceholder(variable_name="chat_history"))
            messages.append(("human", question))
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"chat_history": self.chat_history})
        else:
            messages.append(("human", question))
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({})
        
        # 更新对话历史
        if use_history:
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))
            
            # 限制历史长度（保留最近10轮对话）
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
        
        return {
            "answer": answer,
            "context": retrieved_docs,
            "input": question
        }
    
    def get_answer(self, question: str) -> str:
        """
        获取问题答案（简化版）
        
        Args:
            question: 用户问题
            
        Returns:
            答案字符串
        """
        response = self.invoke(question)
        return response["answer"]
    
    def get_answer_with_sources(self, question: str) -> Dict[str, Any]:
        """
        获取答案和来源文档
        
        Args:
            question: 用户问题
            
        Returns:
            包含答案和来源的字典
        """
        response = self.invoke(question)
        
        return {
            "answer": response["answer"],
            "sources": response.get("context", []),
            "question": question
        }
    
    def clear_history(self):
        """清除对话历史"""
        self.chat_history = []
        print("对话历史已清除")
    
    def stream_answer(self, question: str):
        """
        流式回答问题
        
        Args:
            question: 用户问题
            
        Yields:
            答案片段
        """
        # 获取相关文档
        docs = self.retriever.invoke(question)
        
        # 准备上下文
        context = self._format_docs(docs)
        
        # 准备消息
        messages = [
            ("system", self.system_prompt.format(context=context)),
        ]
        
        if self.chat_history:
            messages.append(MessagesPlaceholder(variable_name="chat_history"))
            
        messages.append(("human", question))
        
        prompt = ChatPromptTemplate.from_messages(messages)
        
        # 流式生成
        chain = prompt | self.llm
        
        if self.chat_history:
            response = chain.stream({"chat_history": self.chat_history})
        else:
            response = chain.stream({})
        
        full_answer = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                content = chunk.content
                full_answer += content
                yield content
        
        # 更新历史
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=full_answer))
        
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
