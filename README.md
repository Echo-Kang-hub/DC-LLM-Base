# 🚗 智能汽车知识库问答系统

基于 LangChain + RAG + Streamlit 构建的高颜值智能问答系统。

## ✨ 功能特点

- 🎯 **智能检索**：基于向量相似度的语义检索，精准定位相关知识
- 💡 **上下文理解**：GPT-4 驱动，理解复杂问题，提供准确答案
- ⚡ **快速响应**：优化的检索流程，毫秒级响应用户查询
- 🎨 **高颜值界面**：基于 Streamlit + Streamlit-shadcn-ui 的现代化 UI
- 📚 **向量存储**：支持 FAISS 和 Chroma 两种向量数据库
- 💬 **对话记忆**：支持多轮对话，理解上下文语境

## 🛠️ 技术栈

- **LangChain**: 0.3.24 - LLM 应用开发框架
- **OpenAI GPT-4**: 大语言模型
- **FAISS**: Facebook 的向量相似度搜索库
- **Streamlit**: 快速构建数据应用的框架
- **Python**: 3.11

## 📦 安装部署

### 1. 环境准备

确保已安装 Python 3.11 和 uv 包管理工具。

### 2. 克隆项目

```bash
git clone <your-repo-url>
cd llm_rag
```

### 3. 激活虚拟环境

```bash
.\.venv\Scripts\activate
```

### 4. 安装依赖

```bash
uv pip install -r requirements.txt
```

### 5. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写你的 OpenAI API Key：

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### 6. 准备知识库

将你的 PDF 知识库文件放在项目根目录，默认文件名为 `car_corpus.pdf`。

### 7. 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中自动打开，默认地址：http://localhost:8501

## 📖 使用指南

### 初次使用

1. 启动应用后，点击左侧边栏的 **"🚀 初始化知识库"** 按钮
2. 系统会自动处理 PDF 文档并创建向量数据库（首次需要几分钟）
3. 初始化完成后即可开始提问

### 提问方式

- **直接输入**：在输入框中输入问题，点击发送按钮
- **测试问题**：从左侧边栏选择预设的测试问题

### 查看来源

每次回答后，可以点击 **"📚 查看参考来源"** 查看答案的来源文档片段。

### 清除历史

点击左侧边栏的 **"🗑️ 清除对话历史"** 可以清空当前对话。

## 🏗️ 项目结构

```
llm_rag/
├── app.py                      # Streamlit 主应用
├── config.py                   # 配置管理
├── document_processor.py       # 文档处理模块
├── vector_store_manager.py     # 向量存储管理
├── rag_chain.py               # RAG 链实现
├── requirements.txt           # 依赖列表
├── .env.example              # 环境变量模板
├── .gitignore                # Git 忽略文件
├── car_corpus.pdf            # 知识库文件
├── test_question.json        # 测试问题
└── vector_store/             # 向量数据库存储目录（自动生成）
```

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| OPENAI_API_KEY | OpenAI API 密钥 | 必填 |
| OPENAI_API_BASE | OpenAI API 基础URL | https://api.openai.com/v1 |
| OPENAI_MODEL | 使用的模型 | gpt-4o-mini |
| EMBEDDING_MODEL | 嵌入模型 | text-embedding-3-small |
| CHUNK_SIZE | 文本分块大小 | 1000 |
| CHUNK_OVERLAP | 文本块重叠大小 | 200 |
| RETRIEVAL_K | 检索文档数量 | 4 |
| VECTOR_STORE_TYPE | 向量存储类型 | faiss |
| TEMPERATURE | 模型温度参数 | 0.7 |

### 向量存储选择

支持两种向量存储：

- **FAISS** (推荐)：速度快，轻量级，适合中小型知识库
- **Chroma**：功能丰富，支持更多高级特性

修改 `.env` 中的 `VECTOR_STORE_TYPE` 即可切换。

## 🔧 高级功能

### 自定义提示词

编辑 `rag_chain.py` 中的 `system_prompt` 可以自定义系统提示词。

### 调整检索参数

修改 `.env` 中的以下参数优化检索效果：

- `CHUNK_SIZE`: 增大可保留更多上下文，减小可提高检索精度
- `CHUNK_OVERLAP`: 增大可减少边界信息丢失
- `RETRIEVAL_K`: 增大可获取更多相关文档，但可能引入噪音

### 更换 LLM 模型

支持所有 OpenAI 兼容的模型，修改 `OPENAI_MODEL` 即可：

- `gpt-4o-mini`: 经济实用
- `gpt-4o`: 性能强大
- `gpt-4-turbo`: 平衡选择

## 🐛 常见问题

### 1. 初始化失败

**问题**：点击初始化按钮后报错

**解决**：
- 检查 `.env` 文件是否正确配置
- 确认 OpenAI API Key 有效且有足够余额
- 检查 `car_corpus.pdf` 文件是否存在

### 2. 回答不准确

**问题**：系统回答不够准确或相关性差

**解决**：
- 增大 `RETRIEVAL_K` 值获取更多上下文
- 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`
- 检查知识库文档质量

### 3. 响应速度慢

**问题**：回答生成时间过长

**解决**：
- 使用 `gpt-4o-mini` 等更快的模型
- 减小 `RETRIEVAL_K` 值
- 考虑使用本地 Embedding 模型

## 📝 开发计划

- [ ] 支持多文件上传
- [ ] 添加文档管理界面
- [ ] 支持更多文档格式（Word、Excel 等）
- [ ] 添加用户反馈机制
- [ ] 支持本地 LLM 模型
- [ ] 添加知识库更新功能
- [ ] 支持多语言

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请提交 Issue。

---

**Powered by LangChain + OpenAI + Streamlit** ❤️
