# ✨ RAG 可视化测试系统

基于 LangChain + RAG + Streamlit 构建的 **Gemini 风格**可视化测试平台，提供智能问答和多种实验功能。

## 🌟 核心亮点

- 🎨 **Gemini 风格界面**：参照 Google Gemini 设计的深色主题，简约现代
- 📊 **可视化配置显示**：左侧边栏实时显示所有系统配置（LLM、Embedding、Chunk Size等）
- 🔬 **多实验页面**：引用标注、Chunk Size对比、记忆机制等多种RAG实验
- 💬 **AI智能问答**：完整的RAG问答成品，支持多轮对话
- 📈 **实时对比分析**：可视化对比不同配置的效果
- ⚡ **快速响应**：优化的检索流程，毫秒级响应
- 🔧 **灵活配置**：支持本地/远程Embedding，FAISS/Chroma向量库

## �️ 界面预览

### 主页 - Gemini 风格欢迎页
- ✨ 中央欢迎区域："Hi! 欢迎使用"
- 🚀 快捷功能按钮
- 📊 功能介绍卡片
- 📡 系统状态显示

### AI 问答页面
- 💬 智能对话界面
- 📚 来源文档展示
- 💡 预设快捷问题
- 🔄 多轮对话支持

### 实验页面
- 🔬 **引用标注实验**：测试LLM标注信息来源能力
- 📏 **Chunk Size对比**：对比不同分块大小的效果
- 🧠 **记忆机制对比**：Buffer Memory vs Summary Memory
- 📊 实时结果对比和可视化

### 左侧配置栏
实时显示所有系统配置：
- Python 版本和系统信息
- LLM 配置（模型、Temperature、API Base）
- Embedding 配置（类型、模型、设备）
- 文档处理配置（Chunk Size、Overlap、检索数量k）
- 向量存储配置（类型、路径）

## 🛠️ 技术栈

- **LangChain**: 0.3.24 - LLM 应用开发框架
- **OpenAI GPT**: 大语言模型（支持GPT-4、GPT-4o-mini等）
- **Embedding**: 支持本地（BGE）和远程（OpenAI）
- **FAISS/Chroma**: 向量相似度搜索库
- **Streamlit**: 多页面可视化应用框架
- **Python**: 3.11+

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

**Gemini 风格多页面系统（推荐）：**

```bash
streamlit run app_gemini.py
```

**传统单页面系统：**

```bash
streamlit run app.py
```

应用将在浏览器中自动打开，默认地址：http://localhost:8501

## 📖 使用指南

### 页面导航

**主页（首页）**
- 查看系统概览和功能介绍
- 查看左侧边栏的完整配置信息
- 了解各个功能模块

**💬 AI 问答页面**
1. 点击 "🚀 初始化RAG系统" 按钮
2. 系统自动加载向量数据库（首次需创建）
3. 输入问题或选择预设问题
4. 查看AI回答和参考来源
5. 支持多轮对话

**🔬 实验页面**

**1. 引用标注实验**
- 测试LLM在回答时标注引用来源的能力
- 支持单个测试和批量测试
- 查看每个答案的引用来源文档
- 对比带引用和不带引用的效果

**2. Chunk Size 对比实验**
- 对比不同文档分块大小（256/512/1024）
- 查看chunk数量、响应时间
- 对比不同chunk size的答案质量
- 支持自定义chunk size测试

**3. 记忆机制对比实验**
- 并排对比 Buffer Memory 和 Summary Memory
- 实时查看两种记忆方式的表现
- 观察Token消耗差异
- 测试长对话场景

### 配置查看

所有配置参数实时显示在左侧边栏：
- **系统信息**：Python版本、操作系统
- **LLM配置**：模型名称、Temperature、API地址
- **Embedding配置**：本地/远程、模型名、设备类型
- **文档配置**：Chunk Size、Chunk Overlap、检索数量k
- **向量库配置**：FAISS/Chroma、存储路径

### 初次使用

1. 启动应用后，进入 "💬 AI问答" 页面
2. 点击 **"🚀 初始化RAG系统"** 按钮
3. 系统会自动处理 PDF 文档并创建向量数据库（首次需要几分钟）
4. 初始化完成后即可开始提问或运行实验

## 🏗️ 项目结构

```
llm_rag/
├── app_gemini.py               # Gemini 风格主页（推荐）
├── app.py                      # 传统单页面应用
├── pages/                      # 多页面模块
│   ├── 01_💬_AI问答.py         # AI智能问答页面
│   ├── 02_🔬_实验_引用标注.py   # 引用标注实验
│   ├── 03_📏_实验_Chunk_Size.py # Chunk Size对比实验
│   └── 04_🧠_实验_记忆机制.py   # 记忆机制对比实验
├── config.py                   # 配置管理
├── document_processor.py       # 文档处理模块
├── vector_store_manager.py     # 向量存储管理
├── rag_chain.py               # RAG 链实现
├── experiment_citation.py      # 引用标注实验（命令行版本）
├── experiment_memory.py        # 记忆机制实验（命令行版本）
├── experiments.py             # 批量实验脚本
├── init_kb.py                 # 知识库初始化脚本
├── main.py                    # 命令行交互入口
├── requirements.txt           # 依赖列表
├── pyproject.toml            # 项目配置
├── .env.example              # 环境变量模板
├── .gitignore                # Git 忽略文件
├── car_corpus.pdf            # 知识库文件
├── test_question.json        # 测试问题集
├── vector_store/             # 向量数据库存储目录
│   └── faiss_index/          # FAISS 索引文件
└── 文档/                      # 项目文档
    ├── 官方文档.md
    ├── INSTALL.md
    ├── QUICKSTART.md
    └── USAGE.md
```

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| OPENAI_API_KEY | OpenAI API 密钥 | 必填 |
| OPENAI_API_BASE | OpenAI API 基础URL | https://api.openai.com/v1 |
| OPENAI_MODEL | 使用的模型 | gpt-4o-mini |
| USE_LOCAL_EMBEDDING | 使用本地Embedding | true |
| EMBEDDING_MODEL | OpenAI Embedding模型 | text-embedding-3-small |
| LOCAL_EMBEDDING_MODEL | 本地Embedding模型 | BAAI/bge-small-zh-v1.5 |
| EMBEDDING_DEVICE | Embedding设备 | cpu (或 cuda) |
| CHUNK_SIZE | 文本分块大小 | 1000 |
| CHUNK_OVERLAP | 文本块重叠大小 | 200 |
| RETRIEVAL_K | 检索文档数量 | 4 |
| VECTOR_STORE_TYPE | 向量存储类型 | faiss (或 chroma) |
| VECTOR_STORE_PATH | 向量存储路径 | ./vector_store |
| TEMPERATURE | 模型温度参数 | 0.7 |
| KNOWLEDGE_BASE_PATH | 知识库PDF路径 | ./car_corpus.pdf |

### Embedding 选择

**本地 Embedding（推荐，免费）**
```env
USE_LOCAL_EMBEDDING=true
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu  # 有显卡可设为 cuda
```

**远程 Embedding（OpenAI）**
```env
USE_LOCAL_EMBEDDING=false
EMBEDDING_MODEL=text-embedding-3-small
```

### 向量存储选择

支持两种向量存储：

- **FAISS** (推荐)：速度快，轻量级，适合中小型知识库
- **Chroma**：功能丰富，支持更多高级特性

修改 `.env` 中的 `VECTOR_STORE_TYPE` 即可切换。

### 实时配置查看

启动应用后，所有配置参数均显示在左侧边栏，无需查看配置文件。

## 🔧 高级功能

### 🔬 实验功能

**1. 引用标注实验**

测试LLM是否能正确标注信息来源，提高答案可追溯性。

```bash
# 命令行版本
python experiment_citation.py

# 交互式模式
python experiment_citation.py interactive
```

Web界面：访问 "🔬 实验_引用标注" 页面

**2. Chunk Size 对比实验**

对比不同文档分块大小对检索和回答质量的影响。

```bash
# 运行批量实验
python experiments.py
```

Web界面：访问 "📏 实验_Chunk Size" 页面

**3. 记忆机制对比实验**

对比 Buffer Memory 和 Summary Memory 在多轮对话中的表现。

```bash
# 命令行版本
python experiment_memory.py
```

Web界面：访问 "🧠 实验_记忆机制" 页面

### 💬 命令行使用

除了Web界面，还支持命令行交互：

```bash
# 基础问答
python main.py

# 初始化知识库
python init_kb.py
```

### 🎨 自定义提示词

编辑 `rag_chain.py` 中的 `system_prompt` 可以自定义系统提示词。

### 📊 调整检索参数

修改 `.env` 中的以下参数优化检索效果：

- `CHUNK_SIZE`: 增大可保留更多上下文，减小可提高检索精度
- `CHUNK_OVERLAP`: 增大可减少边界信息丢失
- `RETRIEVAL_K`: 增大可获取更多相关文档，但可能引入噪音

**推荐配置：**
- 短问答：`CHUNK_SIZE=512, RETRIEVAL_K=3`
- 长文本理解：`CHUNK_SIZE=1024, RETRIEVAL_K=5`
- 精确检索：`CHUNK_SIZE=256, RETRIEVAL_K=5`

### 🤖 更换 LLM 模型

支持所有 OpenAI 兼容的模型，修改 `OPENAI_MODEL` 即可：

- `gpt-4o-mini`: 经济实用，速度快
- `gpt-4o`: 性能强大，理解能力强
- `gpt-4-turbo`: 平衡选择
- 自定义模型：支持任何兼容OpenAI API的模型

## 🐛 常见问题

### 1. 初始化失败

**问题**：点击初始化按钮后报错

**解决**：
- 检查 `.env` 文件是否正确配置
- 确认 OpenAI API Key 有效且有足够余额
- 检查 `car_corpus.pdf` 文件是否存在
- 如使用本地Embedding，确保模型已下载

### 2. 回答不准确

**问题**：系统回答不够准确或相关性差

**解决**：
- 增大 `RETRIEVAL_K` 值获取更多上下文
- 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`
- 检查知识库文档质量
- 在 Chunk Size 实验页面测试最佳配置

### 3. 响应速度慢

**问题**：回答生成时间过长

**解决**：
- 使用 `gpt-4o-mini` 等更快的模型
- 减小 `RETRIEVAL_K` 值
- 使用本地 Embedding 模型
- 检查网络连接

### 4. Streamlit 端口被占用

**问题**：`Port 8501 is not available`

**解决**：
```bash
# 使用其他端口
streamlit run app_gemini.py --server.port 8502
```

### 5. 本地 Embedding 模型下载失败

**问题**：首次使用本地Embedding时下载慢或失败

**解决**：
- 使用镜像源：设置 `HF_ENDPOINT=https://hf-mirror.com`
- 或切换为远程Embedding：`USE_LOCAL_EMBEDDING=false`

## 🎯 使用场景

- **📚 企业知识库问答**：内部文档、手册、政策查询
- **🎓 教育培训**：课程资料、学习材料问答
- **🔬 研究实验**：测试不同RAG配置的效果
- **💼 客服系统**：产品说明、FAQ自动回答
- **📖 文档助手**：大型文档快速查询和理解

## 🧪 实验结果示例

### Chunk Size 对比

| Chunk Size | Chunks数量 | 响应时间 | 答案质量 |
|-----------|----------|---------|---------|
| 256 | 约500个 | 1.2s | 精确但可能缺乏上下文 |
| 512 | 约250个 | 1.5s | **平衡推荐** |
| 1024 | 约125个 | 2.1s | 上下文丰富但可能有噪音 |

### 记忆机制对比

| 类型 | Token消耗 | 适用场景 |
|------|----------|---------|
| Buffer Memory | 随对话增长 | 短对话、精确上下文 |
| Summary Memory | 可控制 | 长对话、成本敏感 |

## 📝 开发计划

- [x] Gemini 风格界面
- [x] 多页面实验功能
- [x] 引用标注实验
- [x] Chunk Size 对比实验
- [x] 记忆机制对比实验
- [x] 配置可视化显示
- [ ] 查询改写实验
- [ ] 重排序实验
- [ ] 混合检索实验
- [ ] 支持多文件上传
- [ ] 添加文档管理界面
- [ ] 支持更多文档格式（Word、Excel 等）
- [ ] 添加用户反馈机制
- [ ] 支持本地 LLM 模型（Ollama）
- [ ] 添加知识库更新功能
- [ ] 导出实验报告功能

## 📄 许可证

MIT License

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd llm_rag

# 2. 激活虚拟环境
.\.venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填写 OPENAI_API_KEY

# 5. 启动应用
streamlit run app_gemini.py

# 6. 访问 http://localhost:8501
```

## 🎬 演示视频

（待添加）

## 📸 界面截图

（待添加）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📧 联系方式

如有问题或建议，请提交 Issue。

## 🌟 Star History

如果这个项目对你有帮助，请给一个 Star ⭐

---

**✨ Powered by LangChain + OpenAI + Streamlit**

**🎨 Inspired by Google Gemini Design**
