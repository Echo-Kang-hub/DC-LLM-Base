"""
Gemini风格的RAG可视化测试系统 - 主页面
"""
import streamlit as st
import sys
import platform
from config import Config

# 页面配置
st.set_page_config(
    page_title="RAG 可视化测试系统",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gemini风格CSS
st.markdown("""
<style>
    /* 全局主题 - Gemini深色风格 */
    :root {
        --primary-color: #8ab4f8;
        --accent-color: #aecbfa;
        --background-dark: #0d1117;
        --background-surface: #161b22;
        --background-hover: #21262d;
        --text-primary: #e6edf3;
        --text-secondary: #7d8590;
        --border-color: #30363d;
        --shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    /* 主容器 */
    .main {
        background-color: var(--background-dark);
    }
    
    /* 隐藏默认装饰 */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    
    /* 欢迎容器 - Gemini风格 */
    .welcome-container {
        text-align: center;
        padding: 6rem 2rem 3rem 2rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .gemini-icon {
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .welcome-title {
        font-size: 3rem;
        font-weight: 400;
        color: var(--text-primary);
        margin: 1rem 0;
        letter-spacing: -1px;
    }
    
    .welcome-subtitle {
        font-size: 2.5rem;
        font-weight: 300;
        color: var(--text-secondary);
        margin: 0.5rem 0 3rem 0;
    }
    
    /* 快捷按钮组 */
    .quick-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: center;
        margin: 3rem 0;
    }
    
    .action-chip {
        background-color: var(--background-surface);
        color: var(--text-primary);
        padding: 0.75rem 1.5rem;
        border-radius: 24px;
        border: 1px solid var(--border-color);
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.95rem;
    }
    
    .action-chip:hover {
        background-color: var(--background-hover);
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background-color: var(--background-surface);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary);
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: var(--text-secondary);
    }
    
    /* 配置卡片 */
    .config-card {
        background-color: var(--background-surface);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
    }
    
    .config-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border-color);
    }
    
    .config-item:last-child {
        border-bottom: none;
    }
    
    .config-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    .config-value {
        color: var(--primary-color);
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    /* 功能卡片 */
    .feature-card {
        background: linear-gradient(135deg, var(--background-surface) 0%, var(--background-hover) 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-4px);
        box-shadow: var(--shadow);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    .feature-desc {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* 按钮 */
    .stButton>button {
        background-color: var(--primary-color);
        color: var(--background-dark);
        border: none;
        border-radius: 24px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: none;
    }
    
    .stButton>button:hover {
        background-color: var(--accent-color);
        transform: scale(1.05);
    }
    
    /* 输入框样式 */
    .stTextInput>div>div>input {
        background-color: var(--background-surface);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        color: var(--text-primary);
        padding: 1rem 1.5rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(138, 180, 248, 0.2);
    }
    
    /* 滚动条 */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    
    /* 状态标签 */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-success {
        background-color: rgba(46, 160, 67, 0.2);
        color: #3fb950;
    }
    
    .status-warning {
        background-color: rgba(187, 128, 9, 0.2);
        color: #d29922;
    }
    
    .status-info {
        background-color: rgba(56, 139, 253, 0.2);
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)


def display_sidebar():
    """显示侧边栏配置信息"""
    with st.sidebar:
        st.markdown("### ⚙️ 系统配置")
        
        # 系统信息
        st.markdown(f"""
        <div class="config-card">
            <div class="config-item">
                <span class="config-label">Python版本</span>
                <span class="config-value">{platform.python_version()}</span>
            </div>
            <div class="config-item">
                <span class="config-label">系统</span>
                <span class="config-value">{platform.system()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # LLM配置
        st.markdown("### 🤖 LLM配置")
        st.markdown(f"""
        <div class="config-card">
            <div class="config-item">
                <span class="config-label">模型</span>
                <span class="config-value">{Config.OPENAI_MODEL}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Temperature</span>
                <span class="config-value">{Config.TEMPERATURE}</span>
            </div>
            <div class="config-item">
                <span class="config-label">API Base</span>
                <span class="config-value">{Config.OPENAI_API_BASE.split('/')[-2] if '/' in Config.OPENAI_API_BASE else 'default'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Embedding配置
        st.markdown("### 🔤 Embedding配置")
        embedding_type = "本地" if Config.USE_LOCAL_EMBEDDING else "远程"
        embedding_model = Config.LOCAL_EMBEDDING_MODEL if Config.USE_LOCAL_EMBEDDING else Config.EMBEDDING_MODEL
        
        st.markdown(f"""
        <div class="config-card">
            <div class="config-item">
                <span class="config-label">类型</span>
                <span class="config-value">{embedding_type}</span>
            </div>
            <div class="config-item">
                <span class="config-label">模型</span>
                <span class="config-value">{embedding_model.split('/')[-1]}</span>
            </div>
            {f'''<div class="config-item">
                <span class="config-label">设备</span>
                <span class="config-value">{Config.EMBEDDING_DEVICE}</span>
            </div>''' if Config.USE_LOCAL_EMBEDDING else ''}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 文档处理配置
        st.markdown("### 📄 文档处理配置")
        st.markdown(f"""
        <div class="config-card">
            <div class="config-item">
                <span class="config-label">Chunk Size</span>
                <span class="config-value">{Config.CHUNK_SIZE}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Chunk Overlap</span>
                <span class="config-value">{Config.CHUNK_OVERLAP}</span>
            </div>
            <div class="config-item">
                <span class="config-label">检索数量 (k)</span>
                <span class="config-value">{Config.RETRIEVAL_K}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 向量存储配置
        st.markdown("### 💾 向量存储配置")
        st.markdown(f"""
        <div class="config-card">
            <div class="config-item">
                <span class="config-label">类型</span>
                <span class="config-value">{Config.VECTOR_STORE_TYPE.upper()}</span>
            </div>
            <div class="config-item">
                <span class="config-label">路径</span>
                <span class="config-value">{Config.VECTOR_STORE_PATH.split('/')[-1]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """主页面"""
    display_sidebar()
    
    # 欢迎区域 - Gemini风格
    st.markdown("""
    <div class="welcome-container">
        <div class="gemini-icon">✨</div>
        <h1 class="welcome-title">Hi! 欢迎使用</h1>
        <h2 class="welcome-subtitle">RAG 可视化测试系统</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 快捷功能按钮
    st.markdown("""
    <div class="quick-actions">
        <div class="action-chip">💬 AI问答</div>
        <div class="action-chip">🔬 实验测试</div>
        <div class="action-chip">📊 数据分析</div>
        <div class="action-chip">⚙️ 系统配置</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 功能介绍卡片
    st.markdown("## 🚀 主要功能")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <h3 class="feature-title">AI智能问答</h3>
            <p class="feature-desc">
                基于RAG技术的智能问答系统，支持多轮对话，
                能够准确检索文档并生成专业回答
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h3 class="feature-title">实验测试</h3>
            <p class="feature-desc">
                提供多种RAG实验：引用标注、Chunk Size对比、
                记忆机制测试、查询改写等
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">可视化分析</h3>
            <p class="feature-desc">
                实时查看系统配置、实验结果对比、
                性能指标分析，支持数据导出
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 使用指南
    st.markdown("## 📖 使用指南")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 快速开始
        1. **查看配置** - 左侧边栏显示所有系统配置
        2. **选择功能** - 从左侧菜单选择AI问答或实验页面
        3. **开始使用** - 输入问题或运行实验
        4. **查看结果** - 分析输出和性能指标
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 实验功能
        - **引用标注实验** - 测试LLM标注信息来源的能力
        - **Chunk Size实验** - 对比不同分块大小的效果
        - **记忆机制实验** - Buffer Memory vs Summary Memory
        - **查询改写实验** - 测试查询优化的影响
        """)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 系统状态
    st.markdown("## 📡 系统状态")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🟢</div>
            <div style="color: var(--text-secondary); font-size: 0.875rem;">LLM服务</div>
            <div style="color: var(--primary-color); font-weight: 500;">运行中</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🟢</div>
            <div style="color: var(--text-secondary); font-size: 0.875rem;">Embedding</div>
            <div style="color: var(--primary-color); font-weight: 500;">就绪</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🟢</div>
            <div style="color: var(--text-secondary); font-size: 0.875rem;">向量库</div>
            <div style="color: var(--primary-color); font-weight: 500;">已加载</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">✨</div>
            <div style="color: var(--text-secondary); font-size: 0.875rem;">版本</div>
            <div style="color: var(--primary-color); font-weight: 500;">v1.0</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 页脚
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); padding: 2rem; border-top: 1px solid var(--border-color);">
        <p style="margin: 0.5rem 0; font-size: 0.875rem;">Powered by LangChain + OpenAI + Streamlit</p>
        <p style="margin: 0.5rem 0; font-size: 0.875rem;">✨ RAG 可视化测试系统</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
