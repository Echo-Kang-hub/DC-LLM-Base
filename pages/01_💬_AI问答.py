"""
AI智能问答 - 成品页面
"""
import streamlit as st
from config import Config
from vector_store_manager import VectorStoreManager
from rag_chain import RAGChain
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="AI智能问答", page_icon="💬", layout="wide")

# 引入通用样式
from app_gemini import display_sidebar

# 初始化session state
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


def initialize_rag():
    """初始化RAG系统"""
    if not st.session_state.initialized:
        try:
            with st.spinner("🔧 正在初始化RAG系统..."):
                vector_store_manager = VectorStoreManager()
                vector_store_manager.load_vector_store()
                
                st.session_state.rag_chain = RAGChain(vector_store_manager)
                st.session_state.initialized = True
                st.success("✅ 系统初始化成功！")
                return True
        except Exception as e:
            st.error(f"❌ 初始化失败: {e}")
            return False
    return True


def main():
    display_sidebar()
    
    # 页面标题
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: var(--text-primary); font-size: 2.5rem; margin: 0;">💬 AI智能问答</h1>
        <p style="color: var(--text-secondary); margin-top: 0.5rem;">基于RAG的智能问答系统</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化按钮
    if not st.session_state.initialized:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 初始化RAG系统", use_container_width=True, type="primary"):
                initialize_rag()
                st.rerun()
        
        st.info("👆 点击上方按钮初始化系统后开始使用")
        return
    
    # 控制按钮
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 重新初始化", use_container_width=True):
            st.session_state.initialized = False
            st.session_state.rag_chain = None
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🗑️ 清除历史", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_history()
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 聊天历史显示
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background-color: var(--background-surface); padding: 1rem 1.5rem; 
                     border-radius: 18px; margin: 1rem 0; max-width: 80%; margin-left: auto; 
                     border: 1px solid var(--border-color);">
                    <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.5rem;">👤 您</div>
                    <div style="color: var(--text-primary);">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: transparent; padding: 1rem 1.5rem; 
                     border-radius: 18px; margin: 1rem 0; max-width: 80%;">
                    <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.5rem;">🤖 AI助手</div>
                    <div style="color: var(--text-primary); line-height: 1.6;">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 显示来源文档
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 查看参考来源", expanded=False):
                        for i, doc in enumerate(msg["sources"], 1):
                            st.markdown(f"**来源 {i}:**")
                            st.text(doc.page_content[:200] + "...")
                            if i < len(msg["sources"]):
                                st.markdown("---")
    
    # 输入区域
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 预设问题
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💡 比亚迪海豹的电池容量？", use_container_width=True):
            st.session_state.preset_question = "比亚迪海豹的电池容量是多少？"
    with col2:
        if st.button("💡 理想L9的座位配置？", use_container_width=True):
            st.session_state.preset_question = "理想L9的座位配置是怎样的？"
    with col3:
        if st.button("💡 特斯拉Model Y性能？", use_container_width=True):
            st.session_state.preset_question = "特斯拉Model Y的加速性能如何？"
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 用户输入
    user_input = st.text_input(
        "请输入您的问题：",
        value=st.session_state.get('preset_question', ''),
        placeholder="例如：如何加热座椅？",
        key="user_input",
        label_visibility="collapsed"
    )
    
    # 清除预设问题
    if 'preset_question' in st.session_state:
        del st.session_state.preset_question
    
    # 发送按钮
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        send_button = st.button("📤 发送", use_container_width=True, type="primary")
    
    # 处理用户输入
    if send_button and user_input:
        # 添加用户消息
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # 获取AI回答
        with st.spinner("🤔 AI正在思考..."):
            try:
                response = st.session_state.rag_chain.get_answer_with_sources(user_input)
                
                # 添加AI消息
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", [])
                })
                
            except Exception as e:
                st.error(f"❌ 生成回答时出错: {e}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"抱歉，处理您的问题时出现了错误：{str(e)}"
                })
        
        st.rerun()


if __name__ == "__main__":
    main()
