"""
简单测试脚本 - 快速诊断问题
"""
import sys
import traceback

print("=" * 60)
print("🧪 系统诊断测试")
print("=" * 60)

# 测试1: 导入模块
print("\n1️⃣ 测试模块导入...")
try:
    from config import Config
    from document_processor import DocumentProcessor
    from vector_store_manager import VectorStoreManager
    from rag_chain import RAGChain
    print("   ✅ 所有模块导入成功")
except Exception as e:
    print(f"   ❌ 模块导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试2: 配置验证
print("\n2️⃣ 测试配置...")
try:
    Config.validate()
    print("   ✅ 配置验证成功")
    print(f"   - 模型: {Config.OPENAI_MODEL}")
    print(f"   - 本地Embedding: {Config.USE_LOCAL_EMBEDDING}")
    if Config.USE_LOCAL_EMBEDDING:
        print(f"   - Embedding模型: {Config.LOCAL_EMBEDDING_MODEL}")
except Exception as e:
    print(f"   ⚠️  配置警告: {e}")

# 测试3: 加载向量存储
print("\n3️⃣ 测试向量存储...")
try:
    vector_store_manager = VectorStoreManager()
    vector_store = vector_store_manager.load_vector_store()
    
    if vector_store:
        print("   ✅ 向量存储加载成功")
    else:
        print("   ⚠️  向量存储不存在，请先运行: python init_kb.py")
        sys.exit(0)
except Exception as e:
    print(f"   ❌ 向量存储加载失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试4: 创建RAG链
print("\n4️⃣ 测试RAG链创建...")
try:
    rag_chain = RAGChain(vector_store_manager)
    print("   ✅ RAG链创建成功")
except Exception as e:
    print(f"   ❌ RAG链创建失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试5: 向量检索
print("\n5️⃣ 测试向量检索...")
test_query = "如何加热座椅？"
try:
    docs = vector_store_manager.similarity_search(test_query, k=2)
    print(f"   ✅ 检索成功，找到 {len(docs)} 个相关文档")
    if docs:
        print(f"   - 第一个文档预览: {docs[0].page_content[:100]}...")
except Exception as e:
    print(f"   ❌ 检索失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试6: 问答功能
print("\n6️⃣ 测试问答功能...")
print(f"   问题: {test_query}")
try:
    answer = rag_chain.get_answer(test_query)
    print(f"   ✅ 回答生成成功")
    print(f"\n   答案:\n   {answer}\n")
except Exception as e:
    print(f"   ❌ 回答生成失败!")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    print("\n   详细堆栈:")
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("🎉 所有测试通过！系统运行正常")
print("=" * 60)
print("\n现在可以运行: streamlit run app.py")
