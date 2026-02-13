"""
实验：记忆机制对比
对比 Buffer Memory 和 Summary Memory 在多轮对话中的表现
"""
import streamlit as st
from config import Config
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="记忆机制实验", page_icon="🧠", layout="wide")

from app_gemini import display_sidebar

# 初始化
if 'memory_initialized' not in st.session_state:
    st.session_state.memory_initialized = False
if 'buffer_history' not in st.session_state:
    st.session_state.buffer_history = []
if 'summary_history' not in st.session_state:
    st.session_state.summary_history = []
if 'summary_text' not in st.session_state:
    st.session_state.summary_text = ""


def count_tokens(text):
    """简单的token计数（粗略估算）"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def summarize_history(llm, messages):
    """压缩对话历史为摘要"""
    if not messages:
        return ""
    
    history_text = "\n".join([
        f"{'用户' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in messages
    ])
    
    summary_prompt = f"""请将以下对话历史压缩为简洁的摘要，保留关键信息：

{history_text}

摘要："""
    
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    return response.content


def buffer_memory_chat(llm, user_input, history):
    """Buffer Memory: 保留完整对话历史"""
    messages = history.copy()
    messages.append(HumanMessage(content=user_input))
    
    response = llm.invoke(messages)
    
    messages.append(AIMessage(content=response.content))
    
    return response.content, messages


def summary_memory_chat(llm, user_input, history, summary, max_history=4):
    """Summary Memory: 超出长度后压缩历史"""
    messages = history.copy()
    
    # 如果历史太长，进行摘要
    if len(messages) > max_history:
        # 保留最近的消息，其余压缩为摘要
        old_messages = messages[:-max_history]
        recent_messages = messages[-max_history:]
        
        # 生成新摘要
        if summary:
            old_messages = [HumanMessage(content=f"之前的对话摘要：{summary}")] + old_messages
        
        new_summary = summarize_history(llm, old_messages)
        messages = recent_messages
        summary = new_summary
    
    # 添加新的用户输入
    messages.append(HumanMessage(content=user_input))
    
    # 如果有摘要，添加到开头
    if summary:
        context_messages = [HumanMessage(content=f"对话历史摘要：{summary}")] + messages
    else:
        context_messages = messages
    
    response = llm.invoke(context_messages)
    
    messages.append(AIMessage(content=response.content))
    
    return response.content, messages, summary


def initialize_system():
    """初始化LLM"""
    try:
        llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE
        )
        st.session_state.llm = llm
        st.session_state.memory_initialized = True
        return True
    except Exception as e:
        st.error(f"❌ 初始化失败: {e}")
        return False


def main():
    display_sidebar()
    
    # 页面标题
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: var(--text-primary); font-size: 2.5rem; margin: 0;">🧠 记忆机制对比实验</h1>
        <p style="color: var(--text-secondary); margin-top: 0.5rem;">Buffer Memory vs Summary Memory</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 实验说明
    with st.expander("📖 实验说明", expanded=True):
        st.markdown("""
        ### 实验目的
        对比两种不同的对话记忆机制在多轮对话中的表现。
        
        ### 记忆类型
        
        **1. Buffer Memory（缓冲记忆）**
        - 保留所有对话历史
        - 优点：不丢失任何信息
        - 缺点：Token消耗随对话增长
        
        **2. Summary Memory（摘要记忆）**
        - 超过长度限制后压缩历史为摘要
        - 优点：Token消耗可控
        - 缺点：可能丢失部分细节
        
        ### 使用场景
        - Buffer Memory：适合短对话、需要精确上下文的场景
        - Summary Memory：适合长对话、成本敏感的场景
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 初始化
    if not st.session_state.memory_initialized:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 初始化实验环境", use_container_width=True, type="primary"):
                if initialize_system():
                    st.success("✅ 初始化成功！")
                    st.rerun()
        return
    
    # 控制按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重新初始化", use_container_width=True):
            st.session_state.memory_initialized = False
            st.session_state.buffer_history = []
            st.session_state.summary_history = []
            st.session_state.summary_text = ""
            st.rerun()
    with col2:
        if st.button("🗑️ 清除对话历史", use_container_width=True):
            st.session_state.buffer_history = []
            st.session_state.summary_history = []
            st.session_state.summary_text = ""
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 对比显示区域
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: var(--background-surface); 
             border-radius: 12px; border: 1px solid var(--border-color);">
            <h3 style="color: var(--text-primary); margin: 0;">💾 Buffer Memory</h3>
            <p style="color: var(--text-secondary); font-size: 0.875rem; margin: 0.5rem 0 0 0;">
                完整历史记录
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Buffer Memory 历史
        buffer_container = st.container()
        with buffer_container:
            for msg in st.session_state.buffer_history:
                if isinstance(msg, HumanMessage):
                    st.markdown(f"""
                    <div style="background-color: var(--background-hover); padding: 0.75rem; 
                         border-radius: 12px; margin: 0.5rem 0;">
                        <div style="color: var(--text-secondary); font-size: 0.75rem;">👤 用户</div>
                        <div style="color: var(--text-primary); font-size: 0.9rem;">{msg.content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: var(--background-surface); padding: 0.75rem; 
                         border-radius: 12px; margin: 0.5rem 0; border: 1px solid var(--border-color);">
                        <div style="color: var(--text-secondary); font-size: 0.75rem;">🤖 AI</div>
                        <div style="color: var(--text-primary); font-size: 0.9rem;">{msg.content}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Token统计
        buffer_tokens = sum(count_tokens(msg.content) for msg in st.session_state.buffer_history)
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; margin-top: 1rem; 
             background-color: var(--background-hover); border-radius: 8px;">
            <span style="color: var(--text-secondary); font-size: 0.875rem;">
                📊 约 {buffer_tokens} tokens · {len(st.session_state.buffer_history)} 条消息
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: var(--background-surface); 
             border-radius: 12px; border: 1px solid var(--border-color);">
            <h3 style="color: var(--text-primary); margin: 0;">📝 Summary Memory</h3>
            <p style="color: var(--text-secondary); font-size: 0.875rem; margin: 0.5rem 0 0 0;">
                摘要 + 最近历史
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示摘要
        if st.session_state.summary_text:
            st.markdown(f"""
            <div style="background-color: var(--background-hover); padding: 0.75rem; 
                 border-radius: 12px; margin: 0.5rem 0; border-left: 3px solid var(--primary-color);">
                <div style="color: var(--text-secondary); font-size: 0.75rem;">📋 历史摘要</div>
                <div style="color: var(--text-primary); font-size: 0.85rem; font-style: italic;">
                    {st.session_state.summary_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary Memory 历史
        summary_container = st.container()
        with summary_container:
            for msg in st.session_state.summary_history:
                if isinstance(msg, HumanMessage):
                    st.markdown(f"""
                    <div style="background-color: var(--background-hover); padding: 0.75rem; 
                         border-radius: 12px; margin: 0.5rem 0;">
                        <div style="color: var(--text-secondary); font-size: 0.75rem;">👤 用户</div>
                        <div style="color: var(--text-primary); font-size: 0.9rem;">{msg.content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: var(--background-surface); padding: 0.75rem; 
                         border-radius: 12px; margin: 0.5rem 0; border: 1px solid var(--border-color);">
                        <div style="color: var(--text-secondary); font-size: 0.75rem;">🤖 AI</div>
                        <div style="color: var(--text-primary); font-size: 0.9rem;">{msg.content}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Token统计
        summary_tokens = sum(count_tokens(msg.content) for msg in st.session_state.summary_history)
        if st.session_state.summary_text:
            summary_tokens += count_tokens(st.session_state.summary_text)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; margin-top: 1rem; 
             background-color: var(--background-hover); border-radius: 8px;">
            <span style="color: var(--text-secondary); font-size: 0.875rem;">
                📊 约 {summary_tokens} tokens · {len(st.session_state.summary_history)} 条消息
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # 输入区域
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 开始对话")
    
    # 预设对话
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💡 我叫张三", use_container_width=True):
            st.session_state.preset_input = "我叫张三"
    with col2:
        if st.button("💡 我今年25岁", use_container_width=True):
            st.session_state.preset_input = "我今年25岁"
    with col3:
        if st.button("💡 我叫什么？", use_container_width=True):
            st.session_state.preset_input = "你还记得我叫什么吗？"
    
    user_input = st.text_input(
        "请输入消息：",
        value=st.session_state.get('preset_input', ''),
        placeholder="例如：我叫张三...",
        label_visibility="collapsed"
    )
    
    if 'preset_input' in st.session_state:
        del st.session_state.preset_input
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        send_button = st.button("📤 发送", use_container_width=True, type="primary")
    
    # 处理输入
    if send_button and user_input:
        with st.spinner("🤔 AI正在思考..."):
            try:
                # Buffer Memory
                buffer_response, new_buffer_history = buffer_memory_chat(
                    st.session_state.llm,
                    user_input,
                    st.session_state.buffer_history
                )
                st.session_state.buffer_history = new_buffer_history
                
                # Summary Memory
                summary_response, new_summary_history, new_summary = summary_memory_chat(
                    st.session_state.llm,
                    user_input,
                    st.session_state.summary_history,
                    st.session_state.summary_text,
                    max_history=4
                )
                st.session_state.summary_history = new_summary_history
                st.session_state.summary_text = new_summary
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()
