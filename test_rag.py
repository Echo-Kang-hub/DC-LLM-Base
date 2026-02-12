"""
测试脚本 - 测试RAG系统的基本功能
"""
import json
from config import Config
from vector_store_manager import VectorStoreManager
from rag_chain import RAGChain


def test_basic_qa():
    """测试基本问答功能"""
    print("=" * 60)
    print("🧪 测试基本问答功能")
    print("=" * 60)
    
    try:
        # 验证配置
        Config.validate()
        
        # 加载向量存储
        print("\n📚 加载向量存储...")
        vector_store_manager = VectorStoreManager()
        if not vector_store_manager.load_vector_store():
            print("❌ 请先运行 python init_kb.py 初始化知识库")
            return False
        
        # 创建RAG链
        print("🔗 创建RAG链...")
        rag_chain = RAGChain(vector_store_manager)
        
        # 加载测试问题
        print("\n📝 加载测试问题...")
        with open('test_question.json', 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        # 测试前5个问题
        print("\n" + "=" * 60)
        print("开始测试...")
        print("=" * 60)
        
        for i, item in enumerate(questions[:5], 1):
            question = item['question']
            print(f"\n问题 {i}: {question}")
            print("-" * 60)
            
            try:
                response = rag_chain.get_answer_with_sources(question)
                print(f"回答: {response['answer']}")
                print(f"\n来源文档数量: {len(response.get('sources', []))}")
                
                if response.get('sources'):
                    print("\n参考来源预览:")
                    for j, doc in enumerate(response['sources'][:2], 1):
                        preview = doc.page_content[:100].replace('\n', ' ')
                        print(f"  {j}. {preview}...")
                
            except Exception as e:
                print(f"❌ 回答失败: {e}")
            
            print("\n" + "=" * 60)
        
        print("\n✅ 测试完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval():
    """测试检索功能"""
    print("\n" + "=" * 60)
    print("🔍 测试检索功能")
    print("=" * 60)
    
    try:
        # 加载向量存储
        vector_store_manager = VectorStoreManager()
        if not vector_store_manager.load_vector_store():
            print("❌ 请先运行 python init_kb.py 初始化知识库")
            return False
        
        # 测试查询
        test_query = "如何加热座椅？"
        print(f"\n测试查询: {test_query}")
        print("-" * 60)
        
        results = vector_store_manager.similarity_search(test_query, k=3)
        
        print(f"\n找到 {len(results)} 个相关文档:\n")
        for i, doc in enumerate(results, 1):
            print(f"文档 {i}:")
            print(f"  内容: {doc.page_content[:200]}...")
            print(f"  元数据: {doc.metadata}")
            print()
        
        print("✅ 检索测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 检索测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n🚗 智能汽车知识库问答系统 - 测试工具\n")
    
    # 测试检索
    if not test_retrieval():
        return
    
    # 测试问答
    if not test_basic_qa():
        return
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
