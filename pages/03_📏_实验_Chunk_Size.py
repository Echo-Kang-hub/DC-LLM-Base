"""
实验：Chunk Size对比
测试不同文档分块大小对检索效果的影响
"""
import streamlit as st
from config import Config
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from rag_chain import RAGChain
import time
import pandas as pd

st.set_page_config(page_title="Chunk Size实验", page_icon="📏", layout="wide")

from app_gemini import display_sidebar

# 初始化
if 'chunk_experiments' not in st.session_state:
    st.session_state.chunk_experiments = []


def test_chunk_size(chunk_size, question, retrieval_k=3):
    """测试特定chunk size"""
    try:
        # 重新处理文档
        doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.2)  # 20% overlap
        )
        splits = doc_processor.process_pdf(Config.KNOWLEDGE_BASE_PATH)
        
        # 创建临时向量存储
        vector_store_manager = VectorStoreManager()
        vector_store_manager.create_vector_store(splits)
        
        # 创建RAG链
        rag_chain = RAGChain(vector_store_manager, retrieval_k=retrieval_k)
        
        # 获取答案
        start_time = time.time()
        response = rag_chain.get_answer(question)
        elapsed_time = time.time() - start_time
        
        return {
            "chunk_size": chunk_size,
            "answer": response,
            "time": elapsed_time,
            "num_chunks": len(splits)
        }
        
    except Exception as e:
        return {
            "chunk_size": chunk_size,
            "error": str(e)
        }


def main():
    display_sidebar()
    
    # 页面标题
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: var(--text-primary); font-size: 2.5rem; margin: 0;">📏 Chunk Size 对比实验</h1>
        <p style="color: var(--text-secondary); margin-top: 0.5rem;">测试不同文档分块大小对检索效果的影响</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 实验说明
    with st.expander("📖 实验说明", expanded=True):
        st.markdown("""
        ### 实验目的
        对比不同chunk size对RAG检索和回答质量的影响。
        
        ### 实验变量
        - **Chunk Size**: 文档分块的大小（字符数）
        - **Chunk Overlap**: 分块之间的重叠部分（通常设为chunk size的20%）
        
        ### 常见Chunk Size选择
        - **256**: 小块，适合精确检索
        - **512**: 中等大小，平衡性能和准确度
        - **1024**: 大块，保留更多上下文
        
        ### 评估指标
        - 回答质量
        - 响应时间
        - 生成的chunk数量
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 实验配置
    st.markdown("### ⚙️ 实验配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Chunk size选择
        st.markdown("**Chunk Size 配置**")
        use_preset = st.checkbox("使用预设值 (256, 512, 1024)", value=True)
        
        if use_preset:
            chunk_sizes = [256, 512, 1024]
            st.info(f"将测试: {', '.join(map(str, chunk_sizes))}")
        else:
            custom_sizes = st.text_input(
                "自定义chunk sizes (逗号分隔)",
                value="256, 512, 1024"
            )
            chunk_sizes = [int(s.strip()) for s in custom_sizes.split(',')]
    
    with col2:
        # 测试问题
        st.markdown("**测试问题**")
        test_question = st.text_input(
            "输入测试问题",
            value="比亚迪海豹的电池容量是多少？"
        )
        
        # 检索数量
        retrieval_k = st.slider("检索文档数量 (k)", 1, 10, 3)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 运行实验按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_button = st.button(
            "🚀 运行实验",
            use_container_width=True,
            type="primary",
            disabled=not test_question
        )
    
    # 运行实验
    if run_button:
        st.markdown("---")
        st.markdown("### 🧪 实验进行中...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, chunk_size in enumerate(chunk_sizes):
            status_text.text(f"正在测试 Chunk Size = {chunk_size}... ({idx+1}/{len(chunk_sizes)})")
            
            with st.spinner(f"处理 Chunk Size = {chunk_size}"):
                result = test_chunk_size(chunk_size, test_question, retrieval_k)
                results.append(result)
            
            progress_bar.progress((idx + 1) / len(chunk_sizes))
        
        status_text.text("✅ 实验完成！")
        
        # 保存结果
        st.session_state.chunk_experiments.append({
            "question": test_question,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        time.sleep(1)
        st.rerun()
    
    # 显示结果
    if st.session_state.chunk_experiments:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 实验结果")
        
        for exp_idx, experiment in enumerate(reversed(st.session_state.chunk_experiments), 1):
            with st.expander(
                f"实验 #{len(st.session_state.chunk_experiments) - exp_idx + 1} - {experiment['timestamp']}", 
                expanded=(exp_idx == 1)
            ):
                st.markdown(f"**问题**: {experiment['question']}")
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 创建对比表格
                df_data = []
                for result in experiment['results']:
                    if 'error' not in result:
                        df_data.append({
                            "Chunk Size": result['chunk_size'],
                            "Chunks数量": result['num_chunks'],
                            "响应时间(秒)": f"{result['time']:.2f}",
                        })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # 显示详细答案
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**详细答案对比**")
                    
                    for result in experiment['results']:
                        if 'error' not in result:
                            st.markdown(f"""
                            <div style="background-color: var(--background-surface); padding: 1.5rem; 
                                 border-radius: 12px; margin: 1rem 0; border: 1px solid var(--border-color);">
                                <div style="color: var(--primary-color); font-weight: 500; margin-bottom: 0.5rem;">
                                    📏 Chunk Size = {result['chunk_size']} 
                                    <span style="color: var(--text-secondary); font-size: 0.875rem;">
                                        (共{result['num_chunks']}个chunks, 用时{result['time']:.2f}s)
                                    </span>
                                </div>
                                <div style="color: var(--text-primary); line-height: 1.6; padding: 1rem; 
                                     background-color: var(--background-hover); border-radius: 8px;">
                                    {result['answer']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Chunk Size {result['chunk_size']}: {result['error']}")
        
        # 清除结果按钮
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ 清除所有实验结果", use_container_width=True):
                st.session_state.chunk_experiments = []
                st.rerun()
    
    # 实验建议
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 💡 实验建议")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Chunk Size 太小 (< 256)**
        - ✅ 检索更精确
        - ❌ 可能丢失上下文
        - ❌ Chunks数量多，处理慢
        """)
    
    with col2:
        st.markdown("""
        **Chunk Size 太大 (> 1024)**
        - ✅ 保留完整上下文
        - ❌ 检索不够精确
        - ❌ Token消耗大
        """)


if __name__ == "__main__":
    main()
