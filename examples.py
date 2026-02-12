"""
简单使用示例
"""

# 示例1: 基本问答
def example_basic_qa():
    """基本问答示例"""
    from config import Config
    from vector_store_manager import VectorStoreManager
    from rag_chain import RAGChain
    
    # 加载向量存储
    vector_store_manager = VectorStoreManager()
    vector_store_manager.load_vector_store()
    
    # 创建RAG链
    rag_chain = RAGChain(vector_store_manager)
    
    # 提问
    question = "如何加热座椅？"
    answer = rag_chain.get_answer(question)
    
    print(f"问题: {question}")
    print(f"答案: {answer}")


# 示例2: 获取详细信息
def example_detailed_qa():
    """获取详细信息示例"""
    from vector_store_manager import VectorStoreManager
    from rag_chain import RAGChain
    
    vector_store_manager = VectorStoreManager()
    vector_store_manager.load_vector_store()
    
    rag_chain = RAGChain(vector_store_manager)
    
    question = "中央显示屏如何切换日间和夜间模式？"
    response = rag_chain.get_answer_with_sources(question)
    
    print(f"问题: {question}")
    print(f"答案: {response['answer']}")
    print(f"\n找到 {len(response['sources'])} 个相关文档:")
    for i, doc in enumerate(response['sources'], 1):
        print(f"\n文档 {i}:")
        print(doc.page_content[:200])


# 示例3: 多轮对话
def example_conversation():
    """多轮对话示例"""
    from vector_store_manager import VectorStoreManager
    from rag_chain import RAGChain
    
    vector_store_manager = VectorStoreManager()
    vector_store_manager.load_vector_store()
    
    rag_chain = RAGChain(vector_store_manager)
    
    # 第一轮
    q1 = "如何加热座椅？"
    a1 = rag_chain.get_answer(q1)
    print(f"Q: {q1}")
    print(f"A: {a1}\n")
    
    # 第二轮（会基于前面的对话）
    q2 = "需要等多久？"
    a2 = rag_chain.get_answer(q2)
    print(f"Q: {q2}")
    print(f"A: {a2}")


# 示例4: 流式输出
def example_streaming():
    """流式输出示例"""
    from vector_store_manager import VectorStoreManager
    from rag_chain import RAGChain
    
    vector_store_manager = VectorStoreManager()
    vector_store_manager.load_vector_store()
    
    rag_chain = RAGChain(vector_store_manager)
    
    question = "副仪表台按钮如何操作中央显示屏？"
    print(f"问题: {question}")
    print("答案: ", end='', flush=True)
    
    for chunk in rag_chain.stream_answer(question):
        print(chunk, end='', flush=True)
    
    print()  # 换行


# 示例5: 批量处理
def example_batch_processing():
    """批量处理示例"""
    import json
    from vector_store_manager import VectorStoreManager
    from rag_chain import RAGChain
    
    # 加载测试问题
    with open('test_question.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # 初始化
    vector_store_manager = VectorStoreManager()
    vector_store_manager.load_vector_store()
    rag_chain = RAGChain(vector_store_manager)
    
    # 批量处理
    results = []
    for item in questions[:5]:  # 只处理前5个
        question = item['question']
        answer = rag_chain.get_answer(question)
        
        results.append({
            'question': question,
            'answer': answer
        })
        
        print(f"✓ {question}")
    
    # 保存结果
    with open('batch_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！结果已保存到 batch_results.json")


if __name__ == "__main__":
    print("请先运行: python init_kb.py 初始化知识库\n")
    print("然后取消注释下面的示例运行：\n")
    
    example_basic_qa()
    example_detailed_qa()
    example_conversation()
    example_streaming()
    example_batch_processing()
