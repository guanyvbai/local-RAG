智能工作流 RAG 平台 V3.1

本项目是一个功能强大、可本地化私有部署的智能工作流RAG平台。它通过一个可动态配置的工作流引擎（n8n），将前端用户交互与一个高度优化的RAG（检索增强生成）后端紧密结合，并具备根据用户意图智能调度不同AI工作流的能力。

系统的核心是其先进的“迭代式自我修正”问答逻辑，能够通过自我反思和多轮检索，显著提升复杂问题的回答质量和准确性。

🌟 核心特性

    🤖 智能工作流调度 (New):

        动态工作流选择: 用户可在前端通过下拉菜单，明确选择执行“知识库问答”、“数据库查询(Text-to-SQL)”或“CPE漏洞查询”等不同的工作流。

        n8n 智能调度中心: n8n作为系统的“大脑”，根据用户的选择，将任务精确路由到最合适的处理后端（RAG、SQL或CVE）。

    🧠 高级RAG引擎:

        双层智能路由: 结合“向量相似度快筛 + LLM精准决策”的两层路由机制，为用户问题智能匹配最相关的知识库。

        混合检索与重排序: 采用 BM25 稀疏检索和向量稠密检索相结合的混合模式，并通过 Cross-Encoder 模型进行重排序，大幅提升检索精度。

        假设性文档嵌入 (HyDE): 为每个文档块生成摘要和可能的问题，创建多向量表示，提升对复杂查询的召回率。

    🔍 多模态数据查询:

        Text-to-SQL: 支持将自然语言问题直接转换为SQL查询，并从连接的MySQL数据库中获取答案。

        CPE漏洞查询: 内置NVD漏洞数据处理能力，可根据标准的CPE（通用平台枚举）字符串精确查询相关漏洞信息。

    📄 深度文档处理:

        语义与结构感知切分: 优先根据文档的标题、段落进行结构化切分，保证文本块的逻辑完整性。

        Excel“一行一向量”: 将Excel的每一行解析为一条独立的、包含表头上下文的文本向量，极大提升了对表格数据的检索精度。

        多格式支持: 支持包括 PDF, DOCX, XLSX, Markdown, JSON, TXT 在内的多种主流文件格式。

    🚀 企业级架构:

        全容器化一键部署: 所有核心服务均通过 Docker Compose 进行统一编排和一键部署。

        本地化模型支持: 依赖 Ollama 服务，可本地部署和运行大语言模型与嵌入模型，确保所有数据处理都在私有环境中完成。

        统一配置管理: 所有服务的环境变量均通过 .env 文件进行统一管理，简化了配置和维护。

        用户与对话持久化: 基于MySQL存储用户信息、聊天历史和会话，保障数据不丢失。

🏗️ 系统架构

系统采用微服务架构，由六大核心服务协同工作构成：

    前端交互界面 (frontend): 基于原生 HTML, CSS, 和 JavaScript 构建，提供实时聊天、知识库管理和用户登录注册等功能。

    RAG核心后端 (backend): 基于 Python 和 FastAPI 构建，负责处理API请求、用户认证、知识库管理以及核心的RAG问答逻辑。

    智能工作流引擎 (n8n): 作为请求代理和任务调度中心，根据前端传来的工作流模式，智能分发任务到后端不同的处理逻辑。

    模型服务层 (ollama): 本地化部署和运行大语言模型（LLM）与嵌入模型（Embedding Model）。

    向量数据库 (qdrant): 存储文档和漏洞信息的向量，并执行高效的相似度检索。

    关系型数据库 (mysql): 存储 n8n 的工作流、执行日志，以及RAG系统的用户、聊天记录和会话元数据。

🛠️ 技术栈

类别	技术	用途
后端	Python, FastAPI	API服务, RAG核心逻辑, SQL/CVE处理
前端	HTML, CSS, JavaScript (原生)	用户交互界面
工作流	n8n	任务调度, 请求代理
AI模型	Ollama, hoangquan456/qwen3-nothink:1.7b, BGE-m3:latest (示例)	本地LLM与嵌入模型服务
数据库	Qdrant, MySQL	向量存储, 用户及会话元数据存储
容器化	Docker, Docker Compose	服务编排与一键部署
RAG核心	sentence-transformers (Cross-Encoder), rank-bm25	文本重排序, 稀疏检索
数据处理	unstructured, pandas	多格式文档解析

🚀 快速开始

1. 环境准备

    已安装 Docker 和 Docker Compose。

    已克隆本项目代码到本地。

    （可选）为保证模型加载速度，建议提前在本地运行Ollama并拉取所需模型：
    Bash

    ollama pull hoangquan456/qwen3-nothink:1.7b
    ollama pull bge-m3:latest

2. 配置环境

    在项目根目录下，直接创建一个名为 .env 的文件。

    将以下内容复制到 .env 文件中，并根据您的环境进行修改。请务必修改 MYSQL_ROOT_PASSWORD 和 MYSQL_PASSWORD 等敏感信息！
    代码段

    # .env

    # =====================
    # 通用配置
    # =====================
    PROJECT_NAME=rag-n8n
    ENV=development

    # =====================
    # MySQL 数据库 (请修改密码)
    # =====================
    MYSQL_ROOT_PASSWORD='YourStrongRootPassword'
    MYSQL_DATABASE=vul_pass
    APP_DATABASE=app_db
    MYSQL_USER=n8n_user
    MYSQL_PASSWORD='YourStrongUserPassword'
    MYSQL_HOST=mysql
    MYSQL_PORT=3306

    # =====================
    # FastAPI 后端
    # =====================
    SECRET_KEY="your-super-secret-key-for-jwt-!@#$%^"
    # SQLAlchemy 连接字符串
    DATABASE_URL=mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DATABASE}
    APP_DATABASE_URL=mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MYSQL_HOST}:${MYSQL_PORT}/${APP_DATABASE}

    # =====================
    # Qdrant 向量数据库
    # =====================
    QDRANT_URL=http://qdrant:6333

    # =====================
    # Ollama 模型服务
    # =====================
    OLLAMA_URL=http://ollama:11434
    EMBEDDING_MODEL=bge-m3:latest
    LLM_MODEL=hoangquan456/qwen3-nothink:1.7b

    # =====================
    # n8n 配置
    # =====================
    N8N_HOST=n8n
    N8N_PORT=5678
    N8N_EDITOR_BASE_URL=http://localhost:5678
    GENERIC_TIMEZONE=Asia/Shanghai
    N8N_CORS_ALLOWED_ORIGINS=*

    # =====================
    # Text-to-SQL 配置
    # =====================
    # 指定Text-to-SQL功能可以访问的数据库表，用逗号分隔。
    # 如果留空，则程序会自动获取数据库中的所有非系统表。
    SQL_INCLUDED_TABLES=assets_cpe

3. 构建与启动

在项目根目录下，执行以下命令：
Bash

# 构建并后台启动所有服务
docker-compose up -d --build

服务在首次启动时，backend 服务会等待 ollama 服务就绪。如果您没有提前拉取模型，Ollama容器会自行下载，这可能需要一些时间，请耐心等待。

4. 访问应用

    前端应用: http://localhost:8000

    n8n 工作流编辑器: http://localhost:5678

5. 使用说明

    注册与登录: 首次使用请在 http://localhost:8000/register 注册账户。系统已内置默认管理员账户 admin / admin，可直接登录。

    管理知识库: 访问 http://localhost:8000/documents。

        创建集合: 输入新集合名称并点击“创建”。

        上传文档: 从下拉菜单选择一个集合，然后点击“选择并上传文件”。

    开始问答:

        返回主聊天界面 http://localhost:8000。

        在输入框上方，从“选择业务场景”下拉菜单中选择您希望执行的任务（如“知识库问答”、“数据库查询”等）。

        在输入框中提问，系统将根据您选择的模式，自动执行相应的后台工作流并返回答案。