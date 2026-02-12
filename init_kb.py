"""
初始化脚本 - 用于第一次运行时初始化知识库
"""
import os
import sys
from config import Config
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager


def check_environment():
    """检查环境配置"""
    print("=" * 60)
    print("🔍 检查环境配置...")
    print("=" * 60)
    
    # 检查.env文件
    if not os.path.exists('.env'):
        print("❌ 未找到 .env 文件")
        print("📝 请先复制 .env.example 为 .env 并填写配置")
        print("   命令: cp .env.example .env")
        return False
    
    # 检查API Key
    try:
        Config.validate()
        print("✅ 环境变量配置正确")
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        return False
    
    # 检查知识库文件
    if not os.path.exists(Config.KNOWLEDGE_BASE_PATH):
        print(f"❌ 知识库文件不存在: {Config.KNOWLEDGE_BASE_PATH}")
        return False
    else:
        print(f"✅ 找到知识库文件: {Config.KNOWLEDGE_BASE_PATH}")
    
    return True


def initialize_vector_store():
    """初始化向量存储"""
    print("\n" + "=" * 60)
    print("🚀 开始初始化向量存储...")
    print("=" * 60)
    
    try:
        # 文档处理
        print("\n📄 步骤 1/3: 处理PDF文档")
        doc_processor = DocumentProcessor()
        splits = doc_processor.process_pdf(Config.KNOWLEDGE_BASE_PATH)
        
        # 创建向量存储
        print(f"\n💾 步骤 2/3: 创建 {Config.VECTOR_STORE_TYPE.upper()} 向量存储")
        vector_store_manager = VectorStoreManager()
        vector_store_manager.create_vector_store(splits)
        
        # 保存向量存储
        print("\n💿 步骤 3/3: 保存向量存储到磁盘")
        vector_store_manager.save_vector_store()
        
        print("\n" + "=" * 60)
        print("✅ 初始化完成！")
        print("=" * 60)
        print(f"📊 统计信息:")
        print(f"   - 文档块数量: {len(splits)}")
        print(f"   - 向量存储类型: {Config.VECTOR_STORE_TYPE}")
        print(f"   - 存储路径: {Config.VECTOR_STORE_PATH}")
        print("\n🎉 现在可以运行应用了:")
        print("   streamlit run app.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n🚗 智能汽车知识库问答系统 - 初始化工具\n")
    
    # 检查环境
    if not check_environment():
        print("\n⚠️  请先修复上述问题后再运行此脚本")
        sys.exit(1)
    
    # 检查是否已经初始化
    if Config.VECTOR_STORE_TYPE.lower() == "faiss":
        save_path = os.path.join(Config.VECTOR_STORE_PATH, "faiss_index")
        if os.path.exists(save_path):
            print("\n⚠️  检测到已存在的向量存储")
            response = input("是否要重新初始化？这将删除现有数据 (y/N): ")
            if response.lower() != 'y':
                print("取消初始化")
                sys.exit(0)
            else:
                import shutil
                shutil.rmtree(Config.VECTOR_STORE_PATH)
                print("已删除旧的向量存储")
    
    # 初始化
    if initialize_vector_store():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
