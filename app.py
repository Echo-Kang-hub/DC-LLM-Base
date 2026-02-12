"""
Streamlit前端应用 - 高颜值的知识库问答界面
"""
import streamlit as st
import os
import json
from typing import Optional
from config import Config
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from rag_chain import RAGChain


# 页面配置
st.set_page_config(
    page_title="智能汽车知识库问答系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - Gemini风格
st.markdown("""
<style>
    /* 全局主题 - Gemini风格 */
    :root {
        --primary-color: #8ab4f8;
        --accent-color: #aecbfa;
        --background-dark: #131314;
        --background-surface: #1e1f20;
        --background-hover: #292a2d;
        --text-primary: #e8eaed;
        --text-secondary: #9aa0a6;
        --border-color: #3c4043;
    }
    
    /* 主容器背景 */
    .main {
        background-color: var(--background-dark);
    }
    
    /* 隐藏默认的顶部装饰 */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    
    /* 标题容器 - 简约风格 */
    .title-container {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
        background-color: transparent;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 400;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* 消息气泡 - Gemini风格 */
    .user-message {
        background-color: var(--background-surface);
        padding: 1rem 1.25rem;
        border-radius: 18px;
        margin: 0.75rem 0;
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        max-width: 85%;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: transparent;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        margin: 0.75rem 0;
        color: var(--text-primary);
        max-width: 85%;
        line-height: 1.6;
    }
    
    /* 按钮样式 - 扁平化 */
    .stButton>button {
        background-color: var(--primary-color);
        color: var(--background-dark);
        border: none;
        border-radius: 20px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: none;
    }
    
    .stButton>button:hover {
        background-color: var(--accent-color);
        box-shadow: none;
    }
    
    .stButton>button:active {
        transform: scale(0.98);
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: var(--background-surface);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* 输入框样式 */
    .stTextInput>div>div>input {
        border-radius: 24px;
        border: 1px solid var(--border-color);
        background-color: var(--background-surface);
        color: var(--text-primary);
        padding: 0.75rem 1.25rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 1px var(--primary-color);
    }
    
    .stTextArea>div>div>textarea {
        border-radius: 16px;
        border: 1px solid var(--border-color);
        background-color: var(--background-surface);
        color: var(--text-primary);
    }
    
    /* 选择框样式 */
    .stSelectbox>div>div {
        background-color: var(--background-surface);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        color: var(--text-primary);
    }
    
    /* 扩展面板 */
    .streamlit-expanderHeader {
        background-color: var(--background-surface);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        color: var(--text-primary);
    }
    
    .streamlit-expanderContent {
        background-color: var(--background-surface);
        border: 1px solid var(--border-color);
        border-top: none;
        border-radius: 0 0 12px 12px;
    }
    
    /* 分割线 */
    hr {
        border-color: var(--border-color);
        margin: 1.5rem 0;
    }
    
    /* 状态指示器 */
    .stSpinner > div {
        border-top-color: var(--primary-color);
    }
    
    /* 信息提示框 */
    .stAlert {
        background-color: var(--background-surface);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        color: var(--text-primary);
    }
    
    /* 成功提示 */
    [data-testid="stSuccess"] {
        background-color: rgba(138, 180, 248, 0.1);
        border-left: 4px solid var(--primary-color);
    }
    
    /* 错误提示 */
    [data-testid="stError"] {
        background-color: rgba(242, 139, 130, 0.1);
        border-left: 4px solid #f28b82;
    }
    
    /* 警告提示 */
    [data-testid="stWarning"] {
        background-color: rgba(251, 188, 4, 0.1);
        border-left: 4px solid #fbc02d;
    }
    
    /* 代码块 */
    code {
        background-color: var(--background-surface);
        color: var(--primary-color);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    
    pre {
        background-color: var(--background-surface);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* 滚动条 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    
    /* 表格样式 */
    table {
        background-color: var(--background-surface);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    thead tr {
        background-color: var(--background-hover);
    }
    
    /* 卡片容器 */
    .stat-card {
        background-color: var(--background-surface);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    .stat-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
    }
    
    /* 标记文本 */
    mark {
        background-color: rgba(138, 180, 248, 0.2);
        color: var(--primary-color);
    }
    
    /* 链接样式 */
    a {
        color: var(--primary-color);
        text-decoration: none;
    }
    
    a:hover {
        color: var(--accent-color);
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


def initialize_system():
    """初始化系统"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.session_state.vector_store_manager = None


def load_test_questions():
    """加载测试问题"""
    try:
        with open('test_question.json', 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return questions
    except Exception as e:
        st.error(f"加载测试问题失败: {e}")
        return []


def setup_knowledge_base():
    """设置知识库"""
    with st.spinner("🔧 正在初始化知识库..."):
        try:
            # 验证配置
            Config.validate()
            
            # 初始化向量存储管理器
            vector_store_manager = VectorStoreManager()
            
            # 尝试加载已有的向量存储
            vector_store = vector_store_manager.load_vector_store()
            
            # 如果不存在，则创建新的
            if vector_store is None:
                st.info("未找到已有向量存储，正在创建新的向量数据库...")
                
                # 处理PDF文档
                doc_processor = DocumentProcessor()
                splits = doc_processor.process_pdf(Config.KNOWLEDGE_BASE_PATH)
                
                # 创建向量存储
                vector_store_manager.create_vector_store(splits)
                vector_store_manager.save_vector_store()
                
                st.success("✅ 向量数据库创建成功！")
            else:
                st.success("✅ 向量数据库加载成功！")
            
            # 创建RAG链
            rag_chain = RAGChain(vector_store_manager)
            
            st.session_state.rag_chain = rag_chain
            st.session_state.vector_store_manager = vector_store_manager
            st.session_state.initialized = True
            
            return True
            
        except Exception as e:
            st.error(f"❌ 初始化失败: {e}")
            return False


def display_chat_message(role: str, content: str):
    """显示聊天消息"""
    if role == "user":
        st.markdown(f'<div class="user-message">👤 {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">🤖 {content}</div>', unsafe_allow_html=True)


def main():
    """主函数"""
    initialize_system()
    
    # 标题区域
    st.markdown("""
    <div class="title-container">
        <h1 class="main-title">🚗 智能汽车知识库问答系统</h1>
        <p class="subtitle">基于 LangChain + RAG 的智能问答助手</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.markdown("## ⚙️ 系统控制")
        
        # 系统状态
        status = "🟢 运行中" if st.session_state.initialized else "🔴 未初始化"
        st.markdown(f"**系统状态:** {status}")
        
        st.markdown("---")
        
        # 初始化按钮
        if not st.session_state.initialized:
            if st.button("🚀 初始化知识库", use_container_width=True):
                success = setup_knowledge_base()
                if success:
                    st.rerun()
        else:
            if st.button("🔄 重新初始化", use_container_width=True):
                st.session_state.initialized = False
                st.session_state.rag_chain = None
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        # 系统配置
        st.markdown("## 📊 系统配置")
        st.markdown(f"**模型:** {Config.OPENAI_MODEL}")
        st.markdown(f"**温度:** {Config.TEMPERATURE}")
        st.markdown(f"**检索数量:** {Config.RETRIEVAL_K}")
        st.markdown(f"**分块大小:** {Config.CHUNK_SIZE}")
        
        st.markdown("---")
        
        # 清除历史
        if st.session_state.initialized and st.session_state.chat_history:
            if st.button("🗑️ 清除对话历史", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.rag_chain:
                    st.session_state.rag_chain.clear_history()
                st.success("对话历史已清除！")
                st.rerun()
        
        st.markdown("---")
        
        # 测试问题
        st.markdown("## 📝 测试问题")
        test_questions = load_test_questions()
        if test_questions:
            selected_q = st.selectbox(
                "选择一个测试问题：",
                [""] + [q["question"] for q in test_questions[:10]],
                key="test_question_select"
            )
            if selected_q and st.button("使用此问题", use_container_width=True):
                st.session_state.selected_test_question = selected_q
    
    # 主界面
    if not st.session_state.initialized:
        # 欢迎界面
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <h3 style="color: var(--primary-color); margin: 0 0 0.5rem 0;">🎯 智能检索</h3>
                <p style="color: var(--text-secondary); margin: 0;">基于向量相似度的语义检索，精准定位相关知识</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-card">
                <h3 style="color: var(--primary-color); margin: 0 0 0.5rem 0;">💡 上下文理解</h3>
                <p style="color: var(--text-secondary); margin: 0;">GPT-4驱动，理解复杂问题，提供准确答案</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-card">
                <h3 style="color: var(--primary-color); margin: 0 0 0.5rem 0;">⚡ 快速响应</h3>
                <p style="color: var(--text-secondary); margin: 0;">优化的检索流程，毫秒级响应用户查询</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👈 请点击左侧 '🚀 初始化知识库' 按钮开始使用")
        
    else:
        # 聊天界面
        st.markdown("## 💬 对话区域")
        
        # 显示聊天历史
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                display_chat_message(message["role"], message["content"])
        
        # 输入区域
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 检查是否有选中的测试问题
        default_question = st.session_state.get('selected_test_question', '')
        if default_question:
            st.session_state.selected_test_question = ''  # 清除标记
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "请输入您的问题：",
                value=default_question,
                placeholder="例如：如何加热座椅？",
                key="user_input"
            )
        
        with col2:
            send_button = st.button("📤 发送", use_container_width=True)
        
        # 处理用户输入
        if send_button and user_input:
            # 检查系统是否已初始化
            if not st.session_state.initialized or not st.session_state.rag_chain:
                st.error("⚠️ 请先点击左侧边栏的 '🚀 初始化知识库' 按钮！")
                st.stop()
            
            # 添加用户消息
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # 获取回答
            with st.spinner("🤔 正在思考..."):
                try:
                    # 调试信息
                    status_placeholder = st.empty()
                    status_placeholder.write("🔍 开始检索相关文档...")
                    
                    response = st.session_state.rag_chain.get_answer_with_sources(user_input)
                    
                    status_placeholder.write("✅ 检索完成，生成答案...")
                    
                    answer = response["answer"]
                    
                    status_placeholder.empty()  # 清除状态信息
                    
                    # 添加助手消息
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    # 显示来源文档
                    if response.get("sources"):
                        with st.expander("📚 查看参考来源"):
                            for i, doc in enumerate(response["sources"], 1):
                                st.markdown(f"**来源 {i}:**")
                                st.text(doc.page_content[:300] + "...")
                                st.markdown("---")
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"❌ 回答生成失败: {e}")
                    with st.expander("查看详细错误信息"):
                        st.code(error_details)
                    
                    # 仍然添加错误消息到历史
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"抱歉，处理您的问题时出现了错误：{str(e)}"
                    })
            
            st.rerun()
    
    # 页脚
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); padding: 2rem; border-top: 1px solid var(--border-color);">
        <p style="margin: 0.5rem 0; font-size: 0.875rem;">Powered by LangChain + OpenAI + Streamlit</p>
        <p style="margin: 0.5rem 0; font-size: 0.875rem;">🚗 智能汽车知识库问答系统 v1.0</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
