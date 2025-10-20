# /* 文件名: backend/config.py, 版本号: 3.3 (最终离线版) */
import os
from dotenv import load_dotenv

load_dotenv()

# --- General Settings ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')

# --- Workflows ---
AVAILABLE_WORKFLOWS = [
    {"value": "rag_query", "name": "知识库问答"},
    {"value": "sql_query", "name": "数据库查询"},
    {"value": "vulnerability_lookup", "name": "CPE漏洞查询"}
]

# --- Vector Database (Qdrant) ---
QDRANT_URL = os.getenv("QDRANT_URL")
AVAILABLE_COLLECTIONS = ["general", "products", "support", "vulnerability"]
QDRANT_COLLECTION_NAME = "general"

# --- RAG Self-Correction ---
MAX_QUERY_RETRY_LOOPS = 2
ROUTING_CONFIDENCE_THRESHOLD = 0.82

# --- 【核心修正】为 Reranker 模型定义唯一的本地加载路径 ---
RERANK_MODEL_PATH = "/app/models/reranker"

# --- Embedding Model (Ollama) Settings ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")

# --- Text Splitter Settings ---
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# --- LLM Settings (Ollama) ---
LLM_PROVIDER = "ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL_NAME = os.getenv("LLM_MODEL")

# --- Conversation Memory ---
CONVERSATION_HISTORY_K = 3

# --- Text-to-SQL ---
SQL_INCLUDED_TABLES_STR = os.getenv("SQL_INCLUDED_TABLES", "")
SQL_INCLUDED_TABLES = [table.strip() for table in SQL_INCLUDED_TABLES_STR.split(',') if table.strip()]

# --- Document ingestion & indexing ---
UPLOAD_STORAGE_DIR = os.getenv("UPLOAD_STORAGE_DIR", "/app/data/uploads")
BM25_REBUILD_THRESHOLD = int(os.getenv("BM25_REBUILD_THRESHOLD", "5"))
BM25_REBUILD_INTERVAL_SECONDS = int(os.getenv("BM25_REBUILD_INTERVAL_SECONDS", "300"))