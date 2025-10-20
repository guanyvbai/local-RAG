# /* 文件名: backend/app.py, 版本号: 3.0 (MySQL 迁移版) */
"""
FastAPI 主应用程序文件。111
【V3.0 更新】:
- 移除了所有 sqlite3 相关的代码。
- 引入了新的 database.py 模块来处理所有用户、对话和消息的数据库操作。
- 所有相关的数据库操作都已重构为使用 SQLAlchemy ORM 和 MySQL。
"""

# 1. 首先进行日志配置
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
# 2. 然后再导入其他所有模块
import os
import io
import re
from fastapi import (
    FastAPI, Depends, UploadFile, File, HTTPException, Request, Form, status, Response
)
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session

# 导入我们自己的模块
import config
import auth
import document_parser
from rag_handler import rag_handler
from sql_query_handler import SQLQueryHandler
from cve_handler import cve_handler
from database import User as DBUser, Chat as DBChat, Message as DBMessage, init_db, get_db

logger = logging.getLogger(__name__)

# --- FastAPI 应用实例和启动事件 ---
app = FastAPI(title="Intelligent RAG & SQL System")

try:
    sql_handler = SQLQueryHandler()
    logger.info("SQLQueryHandler 初始化成功。")
except Exception as e:
    sql_handler = None
    logger.error(f"SQLQueryHandler 初始化失败: {e}", exc_info=True)

@app.on_event("startup")
def on_startup():
    # 使用 database.py 中的函数初始化数据库
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

class SQLQueryRequest(BaseModel):
    question: str

class CPELookupRequest(BaseModel):
    cpe: str

class NVDLoadRequest(BaseModel):
    directory_path: str = "/app/data/nvd_data"

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class RenameChatRequest(BaseModel):
    new_title: str

# --- 认证依赖 ---
async def get_current_active_user(token: str = Depends(auth.oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = auth.decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return User(id=user.id, username=user.username)


# --- API 路由 ---

# --- 认证路由 ---
@app.post("/api/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = auth.create_access_token(data={"sub": user.username})
    return Token(access_token=access_token, token_type="bearer")

@app.post("/api/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = auth.get_password_hash(user.password)
    new_user = DBUser(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully. Please log in."}

# --- 知识库管理路由 ---
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

# --- 聊天与工作流路由 ---
@app.get("/api/workflows", response_model=List[Dict])
async def get_workflows(current_user: User = Depends(get_current_active_user)):
    return config.AVAILABLE_WORKFLOWS

@app.get("/api/chats", response_model=List[Dict])
async def get_chats(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    chats = db.query(DBChat).filter(DBChat.user_id == current_user.id).order_by(DBChat.created_at.desc()).all()
    return [{"id": chat.id, "title": chat.title, "created_at": chat.created_at} for chat in chats]

@app.post("/api/chats", response_model=Dict)
async def create_chat(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    new_chat = DBChat(user_id=current_user.id, title="新的对话")
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return {"id": new_chat.id, "title": new_chat.title}

@app.get("/api/chats/{chat_id}", response_model=List[Dict])
async def get_messages(chat_id: int, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    chat = db.query(DBChat).filter(DBChat.id == chat_id, DBChat.user_id == current_user.id).first()
    if not chat:
        return []
    messages = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp.asc()).all()
    return [{"role": msg.role, "content": msg.content} for msg in messages]

@app.post("/api/chats/{chat_id}/messages")
async def add_message_to_chat(chat_id: int, message: Dict, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    chat = db.query(DBChat).filter(DBChat.id == chat_id, DBChat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    new_message = DBMessage(chat_id=chat_id, role=message['role'], content=message['content'])
    db.add(new_message)
    db.commit()
    return {"status": "success"}

@app.delete("/api/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_endpoint(chat_id: int, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    chat = db.query(DBChat).filter(DBChat.id == chat_id, DBChat.user_id == current_user.id).first()
    if chat:
        db.delete(chat)
        db.commit()

@app.put("/api/chats/{chat_id}/rename")
async def rename_chat_endpoint(chat_id: int, request: RenameChatRequest, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    chat = db.query(DBChat).filter(DBChat.id == chat_id, DBChat.user_id == current_user.id).first()
    if chat:
        chat.title = request.new_title
        db.commit()
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Chat not found")

@app.post("/api/ask")
async def ask(request: AskRequest, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """处理 RAG 知识库问答请求"""
    # Save user message
    user_message = DBMessage(chat_id=request.chat_id, role="user", content=request.question)
    db.add(user_message)
    db.commit()

    messages_from_db = db.query(DBMessage).filter(DBMessage.chat_id == request.chat_id).order_by(DBMessage.timestamp.asc()).all()
    messages = [{"role": msg.role, "content": msg.content} for msg in messages_from_db]

    history = [
        {"user": messages[i]['content'], "assistant": messages[i+1]['content']} 
        for i in range(0, len(messages)-1, 2)
        if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant'
    ]
    history_for_prompt = history[-config.CONVERSATION_HISTORY_K:]

    def stream_generator():
        full_response = ""
        try:
            answer_generator = rag_handler.get_answer(request.question, history_for_prompt)
            for chunk in answer_generator:
                full_response += chunk
                yield chunk
            
            # Save assistant response after stream is complete
            db_session = next(get_db())
            assistant_message = DBMessage(chat_id=request.chat_id, role="assistant", content=full_response)
            db_session.add(assistant_message)
            db_session.commit()
            db_session.close()

        except Exception as e:
            logger.error(f"Error during RAG answer generation: {e}", exc_info=True)
            yield "抱歉，回答时出现错误。"
    
    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.post("/api/sql_query", response_class=Response)
async def sql_query(request: SQLQueryRequest, current_user: User = Depends(get_current_active_user)):
    """处理 Text-to-SQL 请求"""
    if not sql_handler:
        raise HTTPException(status_code=500, detail="SQLQueryHandler 未成功初始化。")
    
    logger.info(f"接收到 SQL 查询请求: '{request.question}'")
    db_schema = sql_handler.get_database_schema()
    generated_sql = sql_handler.generate_sql_query(request.question, db_schema)
    return Response(content=generated_sql, media_type="text/plain")

# --- CPE 漏洞查询路由 ---
@app.post("/api/cve_lookup", response_model=List[Dict[str, Any]])
async def cve_lookup(request: CPELookupRequest, current_user: User = Depends(get_current_active_user)):
    return cve_handler.search_vulnerabilities_by_cpe(request.cpe)

@app.post("/api/load_nvd_data")
async def load_nvd_data(request: NVDLoadRequest, current_user: User = Depends(get_current_active_user)):
    directory_path = request.directory_path
    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=404, detail=f"目录未找到: {directory_path}")
    
    try:
        cve_handler.load_nvd_data_from_directory(directory_path)
        return JSONResponse(
            content={"status": "success", "message": f"NVD数据加载任务已从目录 '{directory_path}' 启动，请关注后端日志。"}, 
            status_code=202
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载NVD数据时出错: {e}")

# --- 前端页面服务 ---
@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend(request: Request, full_path: str):
    base_dir = config.FRONTEND_DIR
    path = full_path.strip() or "index.html"
    
    if ".." in path:
        raise HTTPException(status_code=404, detail="Not Found")

    file_path = os.path.join(base_dir, path)

    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())

    # 兼容 /login, /register 等无 .html 后缀的路径
    if full_path in ["login", "register", "documents"]:
        html_file = os.path.join(base_dir, f'{full_path}.html')
        if os.path.exists(html_file):
            with open(html_file, 'r', encoding='utf-8') as f:
                return HTMLResponse(f.read())

    # 对于 SPA 应用，所有未匹配的路径都应返回 index.html
    index_path = os.path.join(base_dir, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())

    raise HTTPException(status_code=404, detail="Not Found")