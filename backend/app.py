# 文件名: backend/app.py, 版本号: 2.1
"""
FastAPI 主应用程序文件。
负责定义所有 API 路由、处理 HTTP 请求、与数据库交互，
并作为整个后端服务的入口点。
"""

# 1. 首先进行日志配置
import logging
import sys # <--- 导入 sys 模块
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout # <--- 强制输出到标准输出
)
# 2. 然后再导入其他所有模块
import os
import io
import sqlite3
import re
from fastapi import (
    FastAPI, Depends, UploadFile, File, HTTPException, Request, Form, status
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import List, Dict
import time
# 现在导入我们自己的模块
import config
import auth
import document_parser
from rag_handler import rag_handler

# --- 数据库设置 ---
DATA_DIR = "/app/data"
DB_PATH = os.path.join(DATA_DIR, "rag_system.db")
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

# --- 数据库初始化与操作 ---
def init_db():
    """初始化数据库"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, hashed_password TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users (id))")
        cursor.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, chat_id INTEGER, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (chat_id) REFERENCES chats (id))")

        cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        if cursor.fetchone() is None:
            hashed_password = auth.get_password_hash("admin")
            cursor.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", ('admin', hashed_password))
            logger.info("Default user 'admin' with password 'admin' created.")
        conn.commit()

def get_user(username: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()

def create_user(username: str, password: str):
    hashed_password = auth.get_password_hash(password)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
        except sqlite3.IntegrityError:
            return None
    return get_user(username)

def create_new_chat(user_id: int, title: str) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chats (user_id, title) VALUES (?, ?)", (user_id, title))
        conn.commit()
        return cursor.lastrowid

def get_user_chats(user_id: int) -> List[Dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, created_at FROM chats WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        return [dict(row) for row in cursor.fetchall()]

def get_chat_messages(chat_id: int, user_id: int) -> List[Dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
        if cursor.fetchone() is None:
            return []
        cursor.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,))
        return [dict(row) for row in cursor.fetchall()]

def add_message_to_chat(chat_id: int, role: str, content: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, role, content))
        conn.commit()

def delete_chat(chat_id: int, user_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE chat_id = ? AND EXISTS (SELECT 1 FROM chats WHERE id = ? AND user_id = ?)", (chat_id, chat_id, user_id))
        cursor.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
        conn.commit()

def rename_chat(chat_id: int, new_title: str, user_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE chats SET title = ? WHERE id = ? AND user_id = ?", (new_title, chat_id, user_id))
        conn.commit()


# --- FastAPI 应用实例和启动事件 ---
app = FastAPI(title="Intelligent RAG System")

@app.on_event("startup")
def on_startup():
    init_db()


# --- Pydantic 模型定义 ---
class User(BaseModel):
    id: int
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str

class AskRequest(BaseModel):
    question: str
    chat_id: int

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class RenameChatRequest(BaseModel):
    new_title: str


# --- 认证依赖 ---
async def get_current_active_user(token: str = Depends(auth.oauth2_scheme)) -> User:
    payload = auth.decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    user_data = get_user(username=payload.get("sub"))
    if not user_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return User(id=user_data['id'], username=user_data['username'])


# --- API 路由 ---
@app.post("/api/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not auth.verify_password(form_data.password, user['hashed_password']):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = auth.create_access_token(data={"sub": user['username']})
    return Token(access_token=access_token, token_type="bearer")

@app.post("/api/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    if get_user(user.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    if not create_user(user.username, user.password):
        raise HTTPException(status_code=500, detail="Could not create user")
    return {"message": "User created successfully. Please log in."}

@app.get("/api/collections", response_model=List[str])
async def get_collections(current_user: User = Depends(get_current_active_user)):
    return rag_handler.list_collections()

@app.post("/api/collections/{collection_name}", status_code=status.HTTP_201_CREATED)
async def create_collection_endpoint(collection_name: str, current_user: User = Depends(get_current_active_user)):
    if not re.match("^[a-zA-Z0-9_-]+$", collection_name):
        raise HTTPException(status_code=400, detail="集合名称只能包含字母、数字、下划线和连字符。")
    if rag_handler.create_collection(collection_name):
        return {"status": "success", "message": f"集合 '{collection_name}' 创建成功。"}
    raise HTTPException(status_code=500, detail=f"创建集合 '{collection_name}' 失败。")

@app.delete("/api/collections/{collection_name}", status_code=status.HTTP_200_OK)
async def delete_collection_endpoint(collection_name: str, current_user: User = Depends(get_current_active_user)):
    if collection_name == config.QDRANT_COLLECTION_NAME:
         raise HTTPException(status_code=400, detail="不能删除默认集合。")
    if rag_handler.delete_collection(collection_name):
        return {"status": "success", "message": f"集合 '{collection_name}' 删除成功。"}
    raise HTTPException(status_code=500, detail=f"删除集合 '{collection_name}' 失败。")

@app.get("/api/documents", response_model=List[Dict[str, str]])
async def get_documents_endpoint(current_user: User = Depends(get_current_active_user)):
    return rag_handler.list_documents()

@app.delete("/api/documents/{filename}")
async def delete_document_endpoint(filename: str, collection_name: str = Form(...), current_user: User = Depends(get_current_active_user)):
    if rag_handler.delete_document(filename, collection_name): 
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Failed to delete document")

@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...), collection_name: str = Form(...), current_user: User = Depends(get_current_active_user)):
    parser = document_parser.get_parser(file.filename)
    if not parser:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    try:
        content_bytes = await file.read()
        parsed_content = parser(io.BytesIO(content_bytes), file.filename)
        if not parsed_content:
            raise HTTPException(status_code=400, detail="Could not extract text from file.")

        rag_handler.delete_document(file.filename, collection_name)
        rag_handler.process_and_embed_document(parsed_content, file.filename, collection_name)
        
        return {"status": "success", "message": f"File uploaded to {collection_name}"}
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/workflows", response_model=List[Dict])
async def get_workflows(current_user: User = Depends(get_current_active_user)):
    """返回在 config.py 中定义的可用工作流列表。"""
    return config.AVAILABLE_WORKFLOWS

@app.get("/api/chats", response_model=List[Dict])
async def get_chats(current_user: User = Depends(get_current_active_user)):
    return get_user_chats(current_user.id)

@app.post("/api/chats", response_model=Dict)
async def create_chat(current_user: User = Depends(get_current_active_user)):
    chat_id = create_new_chat(current_user.id, "新的对话")
    return {"id": chat_id, "title": "新的对话"}

@app.get("/api/chats/{chat_id}", response_model=List[Dict])
async def get_messages(chat_id: int, current_user: User = Depends(get_current_active_user)):
    return get_chat_messages(chat_id, current_user.id)

@app.delete("/api/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_endpoint(chat_id: int, current_user: User = Depends(get_current_active_user)):
    delete_chat(chat_id, current_user.id)

@app.put("/api/chats/{chat_id}/rename")
async def rename_chat_endpoint(chat_id: int, request: RenameChatRequest, current_user: User = Depends(get_current_active_user)):
    rename_chat(chat_id, request.new_title, current_user.id)
    return {"status": "success"}


@app.post("/api/ask")
async def ask(request: AskRequest, current_user: User = Depends(get_current_active_user)):
    """
    【V2.1 核心修改】
    此接口现在调用 rag_handler.get_answer，该方法内部封装了
    “查询改写 -> 检索 -> 评估 -> 循环/修正”的完整流程。
    """
    """
    【V2.2 诊断版】
    增加了详细的计时日志来排查性能问题。
    """    
    """
    【V2.3 最终诊断版】
    增加了强制的 print 语句来验证代码是否被执行。
    """
    # =================== 最终诊断语句 ===================
    print("--- ASK ENDPOINT ENTERED ---", flush=True)
    # =====================================================

    start_time = time.time()  # <--- 开始计时
    logger.info(f"[TIMING] Received request for chat_id: {request.chat_id}")

    add_message_to_chat(request.chat_id, "user", request.question)
    
    messages = get_chat_messages(request.chat_id, current_user.id)
    history = [
        {"user": messages[i]['content'], "assistant": messages[i+1]['content']} 
        for i in range(0, len(messages)-1, 2)
        if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant'
    ]
    history_for_prompt = history[-config.CONVERSATION_HISTORY_K:]

    def stream_generator():
        full_response = ""
        try:
            # rag_handler.get_answer 现在是一个生成器函数，所以我们需要迭代它
            answer_generator = rag_handler.get_answer(request.question, history_for_prompt)
            for chunk in answer_generator:
                full_response += chunk
                yield chunk
            
            # 当生成器结束时，记录总耗时
            end_time = time.time()
            logger.info(f"[TIMING] Total request duration: {end_time - start_time:.2f} seconds")
            add_message_to_chat(request.chat_id, "assistant", full_response)

        except Exception as e:
            logger.error(f"Error during answer generation: {e}", exc_info=True)
            end_time = time.time()
            logger.info(f"[TIMING] Request failed after: {end_time - start_time:.2f} seconds")
            yield "抱歉，回答时出现错误。"
    
    return StreamingResponse(stream_generator(), media_type="text/plain")


# --- 前端页面服务 ---
@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend(request: Request, full_path: str):
    base_dir = config.FRONTEND_DIR
    path = full_path.strip() or "index.html"
    
    if ".." in path:
        raise HTTPException(status_code=404, detail="Not Found")

    file_path = os.path.join(base_dir, path)

    if os.path.exists(file_path) and os.path.isfile(file_path):
        return HTMLResponse(open(file_path, 'r', encoding='utf-8').read())

    # 兼容 /login, /register 等无 .html 后缀的路径
    if full_path in ["login", "register", "documents"]:
        html_file = os.path.join(base_dir, f'{full_path}.html')
        if os.path.exists(html_file):
            return HTMLResponse(open(html_file, 'r', encoding='utf-8').read())

    # 对于 SPA 应用，所有未匹配的路径都应返回 index.html
    index_path = os.path.join(base_dir, 'index.html')