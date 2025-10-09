# /* 文件名: backend/config.py, 版本号: 2.7 */
"""
项目配置文件。
【V2.7 更新】: 新增了对 SQL_INCLUDED_TABLES 环境变量的读取和解析。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- General Settings ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')


# --- 可用的工作流列表 ---
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

# --- Reranker Model Settings ---
RERANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

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

# --- 【核心新增】Text-to-SQL Settings ---
# 从 .env 读取以逗号分隔的表名字符串
SQL_INCLUDED_TABLES_STR = os.getenv("SQL_INCLUDED_TABLES", "")
# 解析成一个列表，并清除空白字符
SQL_INCLUDED_TABLES = [table.strip() for table in SQL_INCLUDED_TABLES_STR.split(',') if table.strip()]