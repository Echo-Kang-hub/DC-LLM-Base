import json
import time
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

from config import Config
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from rag_chain import RAGChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class RAGExperiments:
    def __init__(self):
        self.knowledge_base_path = Config.KNOWLEDGE_BASE_PATH
        self.test_questions_path = 'test_question.json'
        self.results = []
        
    # 加载测试问题
    def load_test_questions(self, limit=10) -> List[Dict]:
        with open(self.test_questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return questions[:limit]
    
    # ==================== 实验1: Chunk Size对比 ====================
    
    # 实验1：测试不同的Chunk Size对检索效果的影响
    def experiment_chunk_size(self):
        print("\n" + "="*60)
        print("🔬 实验1: Chunk Size对比实验")
        print("="*60)
        
        chunk_sizes = [256,512,1024]
        test_questions = self.load_test_questions(limit=5)
        results = []
        
        for chunk_size in chunk_sizes:
            print(f"\n📏 测试 Chunk Size = {chunk_size}")
            print("-" * 60)
            
            # 重新处理文档
            doc_processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=50
            )
            splits = doc_processor.process_pdf(self.knowledge_base_path)
            
            # 创建向量存储
            vector_store_manager = VectorStoreManager()
            vector_store_manager.create_vector_store(splits)
            
            # 创建RAG链
            rag_chain = RAGChain(vector_store_manager)
            
            # 测试每个问题
            for q in test_questions:
                question = q['question']
                start_time = time.time()
                
                try:
                    response = rag_chain.get_answer_with_sources(question)
                    answer = response['answer']
                    num_sources = len(response['sources'])
                    response_time = time.time() - start_time
                    
                    result = {
                        'chunk_size': chunk_size,
                        'question': question,
                        'answer': answer,
                        'num_sources': num_sources,
                        'response_time': response_time,
                        'num_chunks': len(splits)
                    }
                    results.append(result)
                    
                    print(f"  ✓ {question[:30]}... ({response_time:.2f}s)")
                    
                except Exception as e:
                    print(f"  ✗ {question[:30]}... 失败: {e}")
        
        # 保存结果
        df = pd.DataFrame(results)
        output_file = f'experiment_chunk_size_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 实验完成！结果已保存到: {output_file}")
        self._print_chunk_size_summary(df)
        
        return df
    
    def _print_chunk_size_summary(self, df):
        """打印Chunk Size实验总结"""
        print("\n" + "="*60)
        print("📊 实验结果摘要")
        print("="*60)
        
        summary = df.groupby('chunk_size').agg({
            'response_time': 'mean',
            'num_chunks': 'first'
        }).round(2)
        
        print(summary)
        print(f"\n💡 建议：选择响应时间和准确度平衡的chunk_size")
    
    # ==================== 实验2: Re-ranking (重排序) ====================
    
    def experiment_reranking(self, use_best_chunk_size=True):
        """
        实验2：测试Re-ranking对检索结果的优化
        方法：使用LLM对检索结果进行相关性重排序（采用逐个评分方法）
        
        Args:
            use_best_chunk_size: 是否使用最佳chunk_size=1024重建向量库
        """
        print("\n" + "="*60)
        print("🔬 实验2: Re-ranking (重排序) 实验")
        print("="*60)
        
        # 如果需要使用最佳chunk_size，先重建向量库
        if use_best_chunk_size:
            print("\n📏 使用最佳 Chunk Size = 1024 重建向量库")
            doc_processor = DocumentProcessor(chunk_size=1024, chunk_overlap=200)
            splits = doc_processor.process_pdf(self.knowledge_base_path)
            vector_store_manager = VectorStoreManager()
            vector_store_manager.create_vector_store(splits)
        else:
            # 加载现有向量存储
            vector_store_manager = VectorStoreManager()
            vector_store_manager.load_vector_store()
        
        # 初始化LLM用于重排序
        llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=0,  # 温度=0确保评分稳定
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE
        )
        
        test_questions = self.load_test_questions(limit=5)
        results = []
        
        for q in test_questions:
            question = q['question']
            print(f"\n❓ 问题: {question}")
            print("-" * 60)
            
            # 1. 普通检索 (k=10)
            start_time = time.time()
            docs_original = vector_store_manager.similarity_search(question, k=10)
            time_original = time.time() - start_time
            
            # 2. 使用LLM进行Re-ranking（采用老师的逐个评分方法）
            start_time = time.time()
            docs_reranked = self._rerank_documents_teacher_method(question, docs_original, llm, top_k=3)
            time_reranked = time.time() - start_time
            
            # 3. 分别生成答案
            rag_chain = RAGChain(vector_store_manager)
            
            # 使用原始检索结果
            answer_original = self._get_answer_from_docs(question, docs_original[:4], llm)
            
            # 使用重排序结果
            answer_reranked = self._get_answer_from_docs(question, docs_reranked, llm)
            
            result = {
                'question': question,
                'answer_original': answer_original,
                'answer_reranked': answer_reranked,
                'time_original': time_original,
                'time_reranked': time_reranked,
                'docs_changed': self._docs_order_changed(docs_original[:4], docs_reranked)
            }
            results.append(result)
            
            print(f"  原始检索: {time_original:.2f}s")
            print(f"  重排序后: {time_reranked:.2f}s")
            print(f"  文档顺序变化: {result['docs_changed']}")
        
        # 保存结果
        df = pd.DataFrame(results)
        output_file = f'experiment_reranking_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 实验完成！结果已保存到: {output_file}")
        return df
    
    def _rerank_documents(self, query: str, documents: List, llm, top_k: int = 4) -> List:
        """使用LLM对文档进行重排序"""
        
        # 创建重排序提示
        rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个文档相关性评分专家。请根据用户问题，给每个文档的相关性打分(0-10分)。
只输出JSON格式：[{{"doc_id": 0, "score": 8}}, {{"doc_id": 1, "score": 5}}, ...]"""),
            ("human", """问题: {question}

文档列表:
{documents}

请评分：""")
        ])
        
        # 准备文档列表
        docs_text = ""
        for i, doc in enumerate(documents):
            docs_text += f"\n文档 {i}:\n{doc.page_content[:200]}...\n"
        
        # 调用LLM评分
        try:
            chain = rerank_prompt | llm
            response = chain.invoke({
                "question": query,
                "documents": docs_text
            })
            
            # 解析分数
            import re
            scores_text = response.content
            # 简单的启发式解析（实际应该使用更健壮的方法）
            scores = []
            for i in range(len(documents)):
                # 查找每个文档的分数
                pattern = f'"doc_id":\\s*{i}[^0-9]*"score":\\s*(\\d+)'
                match = re.search(pattern, scores_text)
                if match:
                    scores.append((i, int(match.group(1))))
                else:
                    scores.append((i, 0))
            
            # 按分数排序
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # 返回重排序的文档
            reranked_docs = [documents[doc_id] for doc_id, _ in scores[:top_k]]
            return reranked_docs
            
        except Exception as e:
            print(f"  ⚠️  重排序失败，使用原始顺序: {e}")
            return documents[:top_k]
    
    def _get_answer_from_docs(self, question: str, docs: List, llm) -> str:
        """根据给定文档生成答案"""
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "根据以下上下文回答问题。\n\n上下文:\n{context}"),
            ("human", "{question}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        return response.content
    
    def _docs_order_changed(self, docs1: List, docs2: List) -> bool:
        """检查文档顺序是否改变"""
        if len(docs1) != len(docs2):
            return True
        for i in range(len(docs1)):
            if docs1[i].page_content != docs2[i].page_content:
                return True
        return False
    
    # ==================== 实验3: Query Rewriting (查询改写) ====================
    
    def experiment_query_rewriting(self):
        """
        实验3：测试Query Rewriting对检索效果的提升
        方法：使用LLM改写用户查询，使其更适合检索
        """
        print("\n" + "="*60)
        print("🔬 实验3: Query Rewriting (查询改写) 实验")
        print("="*60)
        
        # 加载向量存储
        vector_store_manager = VectorStoreManager()
        vector_store_manager.load_vector_store()
        
        # 初始化LLM
        llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=0.3,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE
        )
        
        test_questions = self.load_test_questions(limit=5)
        results = []
        
        for q in test_questions:
            original_question = q['question']
            print(f"\n❓ 原始问题: {original_question}")
            print("-" * 60)
            
            # 1. 改写查询
            rewritten_queries = self._rewrite_query(original_question, llm)
            print(f"  改写后:")
            for i, rq in enumerate(rewritten_queries, 1):
                print(f"    {i}. {rq}")
            
            # 2. 使用原始问题检索
            start_time = time.time()
            docs_original = vector_store_manager.similarity_search(original_question, k=5)
            time_original = time.time() - start_time
            answer_original = self._get_answer_from_docs(original_question, docs_original, llm)
            
            # 3. 使用改写后的查询检索（多查询融合）
            start_time = time.time()
            docs_rewritten = self._multi_query_retrieval(
                rewritten_queries, 
                vector_store_manager, 
                k=5
            )
            time_rewritten = time.time() - start_time
            answer_rewritten = self._get_answer_from_docs(original_question, docs_rewritten, llm)
            
            result = {
                'original_question': original_question,
                'rewritten_queries': ' | '.join(rewritten_queries),
                'answer_original': answer_original,
                'answer_rewritten': answer_rewritten,
                'time_original': time_original,
                'time_rewritten': time_rewritten
            }
            results.append(result)
            
            print(f"  原始查询: {time_original:.2f}s")
            print(f"  改写查询: {time_rewritten:.2f}s")
        
        # 保存结果
        df = pd.DataFrame(results)
        output_file = f'experiment_query_rewriting_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 实验完成！结果已保存到: {output_file}")
        return df
    
    def _rewrite_query(self, query: str, llm, num_variants: int = 3) -> List[str]:
        """使用LLM改写查询"""
        
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是一个查询优化专家。请将用户问题改写成{num_variants}个不同的版本，使其更适合语义搜索。

要求：
1. 保持原意
2. 使用不同的表达方式
3. 添加相关关键词
4. 使查询更具体和明确

只输出改写后的问题，每行一个。"""),
            ("human", "{query}")
        ])
        
        chain = rewrite_prompt | llm
        response = chain.invoke({"query": query})
        
        # 解析改写结果
        rewritten = [line.strip() for line in response.content.split('\n') if line.strip()]
        # 移除序号
        rewritten = [q.split('.', 1)[-1].strip() if '.' in q[:5] else q for q in rewritten]
        
        return rewritten[:num_variants]
    
    def _multi_query_retrieval(self, queries: List[str], vector_store_manager, k: int = 4) -> List:
        """多查询融合检索"""
        all_docs = []
        seen_content = set()
        
        # 对每个查询进行检索
        for query in queries:
            docs = vector_store_manager.similarity_search(query, k=k)
            for doc in docs:
                # 去重
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        # 返回前k个
        return all_docs[:k]
    
    # ==================== 主菜单 ====================
    
    def run_menu(self):
        """运行交互式菜单"""
        while True:
            print("\n" + "="*60)
            print("🔬 RAG系统实验平台")
            print("="*60)
            print("\n请选择实验：\n")
            print("  1. 📏 实验1: Chunk Size对比")
            print("  2. 🔄 实验2: Re-ranking (重排序)")
            print("  3. ✍️  实验3: Query Rewriting (查询改写)")
            print("  4. 🎯 运行所有实验")
            print("  5. ❌ 退出")
            print()
            
            choice = input("请输入选项 (1-5): ").strip()
            
            if choice == '1':
                self.experiment_chunk_size()
            elif choice == '2':
                self.experiment_reranking()
            elif choice == '3':
                self.experiment_query_rewriting()
            elif choice == '4':
                print("\n▶️  运行所有实验...")
                self.experiment_chunk_size()
                self.experiment_reranking()
                self.experiment_query_rewriting()
                print("\n🎉 所有实验完成！")
            elif choice == '5':
                print("\n👋 再见！")
                break
            else:
                print("\n❌ 无效的选项，请重新输入")
            
            if choice in ['1', '2', '3', '4']:
                input("\n按 Enter 键继续...")


def main():
    """主函数"""
    experiments = RAGExperiments()
    experiments.run_menu()


if __name__ == "__main__":
    main()
