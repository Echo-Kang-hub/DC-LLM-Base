"""
实验：引用标注
测试LLM在回答时标注引用来源的能力
"""
import streamlit as st
from config import Config
from vector_store_manager import VectorStoreManager
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import time

st.set_page_config(page_title="引用标注实验", page_icon="🔬", layout="wide")

from app_gemini import display_sidebar

# 初始化
if 'citation_initialized' not in st.session_state:
    st.session_state.citation_initialized = False
if 'citation_results' not in st.session_state:
    st.session_state.citation_results = []


def rag_with_citation(query, vectorstore, client):
    """带引用标注的RAG问答"""
    # 检索相关文档
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    
    # 构建带编号的上下文
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[文档{i}] {doc.page_content}")
    context = "\n\n".join(context_parts)
    
    # 带引用要求的Prompt
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
    
    response = client.invoke([HumanMessage(content=prompt)])
    answer = response.content
    
    return answer, retrieved_docs


def initialize_system():
    """初始化系统"""
    try:
        with st.spinner("🔧 正在初始化..."):
            vector_store_manager = VectorStoreManager()
            vector_store_manager.load_vector_store()
            
            client = ChatOpenAI(
                model=Config.OPENAI_MODEL,
                temperature=0,
                openai_api_key=Config.OPENAI_API_KEY,
                openai_api_base=Config.OPENAI_API_BASE
            )
            
            st.session_state.vectorstore = vector_store_manager.vector_store
            st.session_state.llm_client = client
            st.session_state.citation_initialized = True
            
        st.success("✅ 初始化成功！")
        return True
    except Exception as e:
        st.error(f"❌ 初始化失败: {e}")
        return False


def main():
    display_sidebar()
    
    # 页面标题
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: var(--text-primary); font-size: 2.5rem; margin: 0;">🔬 引用标注实验</h1>
        <p style="color: var(--text-secondary); margin-top: 0.5rem;">测试LLM标注信息来源的能力</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 实验说明
    with st.expander("📖 实验说明", expanded=True):
        st.markdown("""
        ### 实验目的
        通过修改Prompt，让LLM在回答时标注每个信息点的来源，提高回答的可追溯性和可信度。
        
        ### 实验方法
        1. 为检索到的每个文档添加编号标记 `[文档1]`, `[文档2]` 等
        2. 在Prompt中明确要求LLM标注引用来源
        3. 对比带引用和不带引用的回答质量
        
        ### 预期效果
        - 回答中每个关键信息都标注了来源文档编号
        - 用户可以快速追溯信息来源
        - 提高系统的透明度和可信度
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 初始化按钮
    if not st.session_state.citation_initialized:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 初始化实验环境", use_container_width=True, type="primary"):
                initialize_system()
                st.rerun()
        return
    
    # 控制按钮
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 重置", use_container_width=True):
            st.session_state.citation_initialized = False
            st.session_state.citation_results = []
            st.rerun()
    with col2:
        if st.button("🗑️ 清除结果", use_container_width=True):
            st.session_state.citation_results = []
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 测试问题
    st.markdown("### 📝 测试问题")
    
    test_questions = [
        "比亚迪海豹的电池容量是多少？",
        "理想L9的座位配置是怎样的？",
        "特斯拉Model Y的加速性能如何？",
        "小鹏P7的智能驾驶功能有哪些？"
    ]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_question = st.text_input(
            "输入自定义问题或选择预设问题：",
            placeholder="例如：比亚迪海豹有哪些配置？"
        )
    with col2:
        selected_preset = st.selectbox(
            "预设问题",
            [""] + test_questions
        )
    
    question_to_test = custom_question if custom_question else selected_preset
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        test_button = st.button("🧪 运行测试", use_container_width=True, type="primary", disabled=not question_to_test)
    
    # 运行测试
    if test_button and question_to_test:
        with st.spinner(f"🔍 正在测试问题: {question_to_test}"):
            try:
                start_time = time.time()
                answer, sources = rag_with_citation(
                    question_to_test,
                    st.session_state.vectorstore,
                    st.session_state.llm_client
                )
                elapsed_time = time.time() - start_time
                
                # 保存结果
                st.session_state.citation_results.append({
                    "question": question_to_test,
                    "answer": answer,
                    "sources": sources,
                    "time": elapsed_time
                })
                
                st.success(f"✅ 测试完成！用时 {elapsed_time:.2f}秒")
                
            except Exception as e:
                st.error(f"❌ 测试失败: {e}")
    
    # 显示结果
    if st.session_state.citation_results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 测试结果")
        
        for idx, result in enumerate(reversed(st.session_state.citation_results), 1):
            with st.container():
                st.markdown(f"""
                <div style="background-color: var(--background-surface); padding: 1.5rem; 
                     border-radius: 16px; margin: 1rem 0; border: 1px solid var(--border-color);">
                    <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.5rem;">
                        测试 #{len(st.session_state.citation_results) - idx + 1} · 用时 {result['time']:.2f}秒
                    </div>
                    <div style="color: var(--primary-color); font-weight: 500; margin-bottom: 1rem;">
                        ❓ {result['question']}
                    </div>
                    <div style="color: var(--text-primary); line-height: 1.6; padding: 1rem; 
                         background-color: var(--background-hover); border-radius: 12px;">
                        {result['answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 显示引用来源
                with st.expander(f"📚 查看引用来源 ({len(result['sources'])}个文档)", expanded=False):
                    for i, doc in enumerate(result['sources'], 1):
                        st.markdown(f"**[文档{i}]**")
                        st.text(doc.page_content[:300] + "...")
                        if i < len(result['sources']):
                            st.markdown("---")
    
    # 批量测试
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 🔄 批量测试")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 运行所有预设问题", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, question in enumerate(test_questions):
                status_text.text(f"正在测试 ({idx+1}/{len(test_questions)}): {question}")
                
                try:
                    start_time = time.time()
                    answer, sources = rag_with_citation(
                        question,
                        st.session_state.vectorstore,
                        st.session_state.llm_client
                    )
                    elapsed_time = time.time() - start_time
                    
                    st.session_state.citation_results.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                        "time": elapsed_time
                    })
                    
                except Exception as e:
                    st.error(f"问题 '{question}' 测试失败: {e}")
                
                progress_bar.progress((idx + 1) / len(test_questions))
                time.sleep(0.5)  # 避免API限流
            
            status_text.text("✅ 批量测试完成！")
            time.sleep(1)
            st.rerun()


if __name__ == "__main__":
    main()
