# 文件名: backend/config.py, 版本号: 2.5
"""
项目配置文件。
集中管理所有硬编码的常量和从环境变量加载的配置。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- General Settings ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')


# --- 【核心新增】可用的工作流列表 ---
# 在这里添加或修改，前端会自动更新
# "value": 将被发送给 n8n Switch 节点的值
# "name": 在前端下拉菜单中显示的名称
AVAILABLE_WORKFLOWS = [
    {"value": "rag_query", "name": "知识库问答"},
    {"value": "sql_query", "name": "数据库查询"},
    {"value": "vulnerability_analysis", "name": "资产漏洞分析"}
]


# --- Vector Database (Qdrant) Settings ---
QDRANT_URL = os.getenv("QDRANT_URL")
AVAILABLE_COLLECTIONS = ["general", "products", "support", "vulnerability"]
QDRANT_COLLECTION_NAME = "general"


# --- RAG Self-Correction Settings ---
MAX_QUERY_RETRY_LOOPS = 2
ROUTING_CONFIDENCE_THRESHOLD = 0.82

# --- 【核心新增】Reranker Model Settings ---
RERANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2" # flashrank 使用的轻量级、高性能模型

# --- Embedding Model Settings ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")


# --- Text Splitter Settings ---
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# --- LLM Settings ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# --- Ollama Settings ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL_NAME = os.getenv("LLM_MODEL")

# --- OpenAI Settings (兼容) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")


# --- Conversation Memory Settings ---
CONVERSATION_HISTORY_K = 3