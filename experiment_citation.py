"""
实验1：带引用标注的RAG问答
通过修改Prompt让LLM在回答时标注引用来源
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config import Config
from vector_store_manager import VectorStoreManager
from document_processor import DocumentProcessor


def rag_with_citation(query, vectorstore, client):
    """带引用标注的RAG问答
    
    Args:
        query: 用户问题
        vectorstore: 向量存储实例
        client: ChatOpenAI客户端
        
    Returns:
        answer: LLM生成的回答
        retrieved_docs: 检索到的文档列表
    """
    # 第1步：检索相关文档
    retrieved_docs = vectorstore.similarity_search(query, k=5)

    # 第2步：构建带编号的上下文
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[文档{i}] {doc.page_content}")
    context = "\n\n".join(context_parts)

    # 第3步：用带引用要求的Prompt生成回答
    prompt = f"""基于以下参考文档回答用户的问题。

要求：
1. 回答中的每个关键信息点都必须标注引用来源，格式为[文档X]
2. 如果某个信息来自多个文档，标注所有相关文档，如[文档1][文档3]
3. 如果参考文档中没有相关信息，明确说明"根据现有资料无法回答"
4. 不要编造参考文档中没有的信息

参考文档：
{context}

用户问题：{query}

请回答（记得标注引用）："""

    # 使用LangChain的invoke方法调用LLM
    response = client.invoke([HumanMessage(content=prompt)])
    answer = response.content

    # 第4步：展示回答和引用来源
    print("=" * 60)
    print(f"问题：{query}\n")
    print(f"回答：{answer}\n")
    print("引用来源：")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  [文档{i}] {doc.page_content[:100]}...")
    print("=" * 60)

    return answer, retrieved_docs


def run_citation_experiment():
    """运行引用标注实验"""
    print("\n" + "="*80)
    print("🔬 实验1: 带引用标注的RAG问答")
    print("="*80)
    
    # 初始化向量存储管理器
    print("\n📚 加载向量存储...")
    vector_store_manager = VectorStoreManager()
    vector_store_manager.load_vector_store()
    vectorstore = vector_store_manager.vector_store
    
    # 初始化LLM客户端
    print("🤖 初始化LLM客户端...")
    client = ChatOpenAI(
        model=Config.OPENAI_MODEL,
        temperature=0,
        openai_api_key=Config.OPENAI_API_KEY,
        openai_api_base=Config.OPENAI_API_BASE
    )
    
    # 测试问题列表
    test_questions = [
        "比亚迪海豹的电池容量是多少？",
        "领克的座位配置是怎样的？",
        "特斯拉Model Y的加速性能如何？",
        "如何通过中央显示屏进行副驾驶员座椅设置？",
        "如何打开车辆尾门？"
    ]
    
    print("\n" + "="*80)
    print("开始测试带引用标注的RAG问答")
    print("="*80 + "\n")
    
    # 对每个问题进行测试
    for idx, question in enumerate(test_questions, 1):
        print(f"\n{'🔸'*40}")
        print(f"测试 {idx}/{len(test_questions)}")
        print(f"{'🔸'*40}\n")
        
        answer, sources = rag_with_citation(question, vectorstore, client)
        
        # 添加延时避免API限流
        if idx < len(test_questions):
            import time
            time.sleep(1)
    
    print("\n" + "="*80)
    print("✅ 实验完成！")
    print("="*80)


def interactive_citation_mode():
    """交互式带引用问答模式"""
    print("\n" + "="*80)
    print("💬 交互式带引用问答模式")
    print("="*80)
    print("提示：输入 'quit' 或 'exit' 退出\n")
    
    # 初始化向量存储管理器
    print("📚 加载向量存储...")
    vector_store_manager = VectorStoreManager()
    vector_store_manager.load_vector_store()
    vectorstore = vector_store_manager.vector_store
    
    # 初始化LLM客户端
    print("🤖 初始化LLM客户端...\n")
    client = ChatOpenAI(
        model=Config.OPENAI_MODEL,
        temperature=0,
        openai_api_key=Config.OPENAI_API_KEY,
        openai_api_base=Config.OPENAI_API_BASE
    )
    
    print("准备就绪！请输入您的问题：\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("👤 您的问题: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n👋 再见！")
                break
            
            # 跳过空输入
            if not user_input:
                continue
            
            # 调用带引用的RAG问答
            print()
            answer, sources = rag_with_citation(user_input, vectorstore, client)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}\n")


if __name__ == "__main__":
    import sys
    
    # 可以通过命令行参数选择模式
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_citation_mode()
    else:
        run_citation_experiment()
