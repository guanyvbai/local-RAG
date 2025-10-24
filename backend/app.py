# /* 文件名: backend/app.py, 版本号: 3.0 (MySQL 迁移版) */
"""backend.app
=================

应用入口点，负责装配 FastAPI 对象、注册依赖与路由，并协调 RAG、Text-to-SQL
以及漏洞查询等子系统。模块级注释主要用于补充项目历史背景，以帮助后来者
快速了解为何存在某些初始化步骤。

【V3.0 更新】
----------------
* 移除了所有 SQLite 相关代码，全部迁移到 MySQL。
* 引入 :mod:`database` 模块集中管理 ORM 模型、会话与初始化逻辑。
* 将遗留的原始 SQL 操作替换为 SQLAlchemy ORM，简化事务管理。
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
    FastAPI, Depends, UploadFile, File, HTTPException, Request, Form, status, Response, BackgroundTasks
)
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from pathlib import Path
import uuid

# 导入我们自己的模块
import config
import auth
import document_parser
from rag_handler import get_rag_handler, get_rag_handler_status
from sql_query_handler import get_sql_query_handler, get_sql_handler_status
from cve_handler import cve_handler
from database import User as DBUser, Chat as DBChat, Message as DBMessage, init_db, get_db

logger = logging.getLogger(__name__)

# --- FastAPI 应用实例和启动事件 ---
app = FastAPI(title="Intelligent RAG & SQL System")

UPLOAD_STORAGE_DIR = Path(config.UPLOAD_STORAGE_DIR)
UPLOAD_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
def on_startup():
    """在 FastAPI 生命周期开始时执行一次数据库初始化。"""

    # 使用 database.py 中的函数初始化数据库
    init_db()
    logger.info("应用启动完成。RAG 与 SQL 处理器将在首次使用时初始化。")


def _require_rag_handler() -> "RAGHandler":
    """懒加载 RAG 处理器，在失败时返回统一的 503 响应。"""

    try:
        return get_rag_handler()
    except Exception as exc:
        logger.error(f"RAGHandler 初始化失败或未就绪: {exc}")
        raise HTTPException(status_code=503, detail="RAG 检索服务暂不可用，请稍后再试。")


def _require_sql_handler() -> "SQLQueryHandler":
    """在访问 Text-to-SQL 能力前验证处理器的就绪状态。"""

    try:
        return get_sql_query_handler()
    except Exception as exc:
        logger.error(f"SQLQueryHandler 初始化失败或未就绪: {exc}")
        raise HTTPException(status_code=503, detail="Text-to-SQL 服务暂不可用，请稍后再试。")

# --- Pydantic 模型定义 ---
class User(BaseModel):
    """经过认证的轻量用户信息。"""

    id: int
    username: str

class Token(BaseModel):
    """标准 OAuth2 Bearer token 响应模型。"""

    access_token: str
    token_type: str

class AskRequest(BaseModel):
    """RAG 对话请求的载荷。"""

    question: str
    chat_id: int

class SQLQueryRequest(BaseModel):
    """Text-to-SQL 查询请求体。"""

    question: str

class CPELookupRequest(BaseModel):
    """CVE 查询接口使用的 CPE 条目。"""

    cpe: str

class NVDLoadRequest(BaseModel):
    """触发 NVD 离线数据加载任务时使用的路径信息。"""

    directory_path: str = "/app/data/nvd_data"

class UserCreate(BaseModel):
    """注册流程使用的表单字段约束。"""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class RenameChatRequest(BaseModel):
    """重命名现有对话时提交的新标题。"""

    new_title: str

# --- 认证依赖 ---
async def get_current_active_user(token: str = Depends(auth.oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """解析访问令牌并返回数据库中的完整用户对象。"""

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
    """通过用户名密码交换访问令牌。"""

    user = db.query(DBUser).filter(DBUser.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = auth.create_access_token(data={"sub": user.username})
    return Token(access_token=access_token, token_type="bearer")

@app.post("/api/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """创建一个新的平台用户。"""

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
    """列出当前可用的向量集合名称。"""

    handler = _require_rag_handler()
    return handler.list_collections()

@app.post("/api/collections/{collection_name}", status_code=status.HTTP_201_CREATED)
async def create_collection_endpoint(collection_name: str, current_user: User = Depends(get_current_active_user)):
    """新建一个供 RAG 使用的向量集合。"""

    if not re.match("^[a-zA-Z0-9_-]+$", collection_name):
        raise HTTPException(status_code=400, detail="集合名称只能包含字母、数字、下划线和连字符。")
    handler = _require_rag_handler()
    if handler.create_collection(collection_name):
        return {"status": "success", "message": f"集合 '{collection_name}' 创建成功。"}
    raise HTTPException(status_code=500, detail=f"创建集合 '{collection_name}' 失败。")

@app.delete("/api/collections/{collection_name}", status_code=status.HTTP_200_OK)
async def delete_collection_endpoint(collection_name: str, current_user: User = Depends(get_current_active_user)):
    """删除指定集合，默认集合除外。"""

    if collection_name == config.QDRANT_COLLECTION_NAME:
         raise HTTPException(status_code=400, detail="不能删除默认集合。")
    handler = _require_rag_handler()
    if handler.delete_collection(collection_name):
        return {"status": "success", "message": f"集合 '{collection_name}' 删除成功。"}
    raise HTTPException(status_code=500, detail=f"删除集合 '{collection_name}' 失败。")

@app.get("/api/documents", response_model=List[Dict[str, str]])
async def get_documents_endpoint(current_user: User = Depends(get_current_active_user)):
    """返回知识库中文档的元信息。"""

    handler = _require_rag_handler()
    return handler.list_documents()

@app.delete("/api/documents/{filename}")
async def delete_document_endpoint(filename: str, collection_name: str = Form(...), current_user: User = Depends(get_current_active_user)):
    """移除集合内的指定文档。"""

    handler = _require_rag_handler()
    if handler.delete_document(filename, collection_name):
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Failed to delete document")


@app.get("/api/system_status", response_model=Dict[str, Dict[str, Any]])
async def system_status(current_user: User = Depends(get_current_active_user)):
    """Expose readiness information for heavy backends."""

    return {
        "rag": get_rag_handler_status(),
        "text_to_sql": get_sql_handler_status(),
    }

def _process_uploaded_document(file_path: Path, original_filename: str, collection_name: str):
    """后台任务：解析、分块并向量化上传的文档。"""

    logger.info(
        "开始后台处理上传文件 '%s'，目标集合 '%s'。临时文件: %s",
        original_filename,
        collection_name,
        file_path,
    )
    parser = document_parser.get_parser(original_filename)
    if not parser:
        logger.error(f"No parser available for file '{original_filename}'.")
        return

    try:
        with file_path.open("rb") as file_handle:
            file_bytes = file_handle.read()
        parsed_content = parser(io.BytesIO(file_bytes), original_filename)
        if not parsed_content:
            logger.error(f"Parsed content empty for '{original_filename}'.")
            return

        parsed_count = len(parsed_content) if isinstance(parsed_content, list) else 'unknown'
        logger.info(
            "文件 '%s' 解析完成，得到 %s 个结构化元素，准备进入分块与向量化流程。",
            original_filename,
            parsed_count,
        )

        try:
            handler = get_rag_handler()
        except Exception as handler_exc:
            logger.error(
                f"RAGHandler 未就绪，无法处理后台文档 '{original_filename}': {handler_exc}"
            )
            return

        logger.info(
            "清理集合 '%s' 中已有的文档 '%s' 并重新向量化。",
            collection_name,
            original_filename,
        )
        handler.delete_document(original_filename, collection_name)
        handler.process_and_embed_document(parsed_content, original_filename, collection_name)
    except Exception as exc:
        logger.error(f"Background processing failed for '{original_filename}': {exc}", exc_info=True)
    finally:
        try:
            file_path.unlink(missing_ok=True)
        except Exception as cleanup_exc:
            logger.warning(f"Failed to remove temporary file '{file_path}': {cleanup_exc}")


async def _save_upload_to_disk(upload: UploadFile, destination: Path) -> int:
    """将上传内容写入磁盘并返回写入字节数。"""

    logger.info("开始保存上传文件 '%s' 到本地路径 %s。", upload.filename, destination)
    total_written = 0
    with destination.open("wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
            total_written += len(chunk)
    await upload.close()
    logger.info(
        "上传文件 '%s' 写入完成，共写入 %.2f MB。",
        upload.filename,
        total_written / (1024 * 1024) if total_written else 0,
    )
    return total_written


@app.post("/api/upload")
async def upload_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """接收用户上传的文件并调度后台解析任务。"""

    logger.info(
        "用户 '%s' 请求上传文件 '%s' 到集合 '%s'。",
        current_user.username,
        file.filename,
        collection_name,
    )
    parser = document_parser.get_parser(file.filename)
    if not parser:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    destination = UPLOAD_STORAGE_DIR / temp_filename

    try:
        total_bytes = await _save_upload_to_disk(file, destination)
    except Exception as exc:
        logger.error(f"Failed to save upload '{file.filename}' to disk: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store uploaded file")

    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    logger.info(
        "文件 '%s' 保存成功 (%.2f MB)，提交至后台任务进行解析与向量化。",
        file.filename,
        total_bytes / (1024 * 1024),
    )
    background_tasks.add_task(_process_uploaded_document, destination, file.filename, collection_name)
    logger.info(
        "文件 '%s' 的后台处理任务已排队，临时文件: %s。",
        file.filename,
        destination,
    )
    return {
        "status": "accepted",
        "message": f"File '{file.filename}' queued for processing in collection '{collection_name}'."
    }

# --- 聊天与工作流路由 ---
@app.get("/api/workflows", response_model=List[Dict])
async def get_workflows(current_user: User = Depends(get_current_active_user)):
    """返回前端可展示的内置工作流列表。"""

    return config.AVAILABLE_WORKFLOWS

@app.get("/api/chats", response_model=List[Dict])
async def get_chats(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """列出当前用户的所有对话元数据。"""

    chats = db.query(DBChat).filter(DBChat.user_id == current_user.id).order_by(DBChat.created_at.desc()).all()
    return [{"id": chat.id, "title": chat.title, "created_at": chat.created_at} for chat in chats]

@app.post("/api/chats", response_model=Dict)
async def create_chat(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """为当前用户创建一条新的对话记录。"""

    new_chat = DBChat(user_id=current_user.id, title="新的对话")
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return {"id": new_chat.id, "title": new_chat.title}

@app.get("/api/chats/{chat_id}", response_model=List[Dict])
async def get_messages(chat_id: int, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """检索指定对话下的所有消息按时间排序。"""

    chat = db.query(DBChat).filter(DBChat.id == chat_id, DBChat.user_id == current_user.id).first()
    if not chat:
        return []
    messages = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp.asc()).all()
    return [{"role": msg.role, "content": msg.content} for msg in messages]

@app.post("/api/chats/{chat_id}/messages")
async def add_message_to_chat(chat_id: int, message: Dict, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """将任意角色的消息持久化到指定对话中。"""

    chat = db.query(DBChat).filter(DBChat.id == chat_id, DBChat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    new_message = DBMessage(chat_id=chat_id, role=message['role'], content=message['content'])
    db.add(new_message)
    db.commit()
    logger.info(
        "用户 '%s' 在对话 %d 中新增一条 %s 消息。",
        current_user.username,
        chat_id,
        message['role'],
    )
    return {"status": "success"}

@app.delete("/api/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_endpoint(chat_id: int, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """软删除策略未启用，因此直接删除整条对话记录。"""

    chat = db.query(DBChat).filter(DBChat.id == chat_id, DBChat.user_id == current_user.id).first()
    if chat:
        db.delete(chat)
        db.commit()
        logger.info("用户 '%s' 删除了对话 %d。", current_user.username, chat_id)

@app.put("/api/chats/{chat_id}/rename")
async def rename_chat_endpoint(chat_id: int, request: RenameChatRequest, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """更新对话标题，供前端列表展示。"""

    chat = db.query(DBChat).filter(DBChat.id == chat_id, DBChat.user_id == current_user.id).first()
    if chat:
        chat.title = request.new_title
        db.commit()
        logger.info(
            "用户 '%s' 将对话 %d 重命名为 '%s'。",
            current_user.username,
            chat_id,
            request.new_title,
        )
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Chat not found")

@app.post("/api/ask")
async def ask(request: AskRequest, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """处理 RAG 知识库问答请求"""
    rag_handler_instance = _require_rag_handler()

    logger.info(
        "用户 '%s' 在对话 %d 提出问题: '%s'。",
        current_user.username,
        request.chat_id,
        request.question if len(request.question) <= 60 else f"{request.question[:57]}...",
    )
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
    logger.info(
        "对话 %d 的历史轮次: %d (截取用于提示的轮次: %d)。",
        request.chat_id,
        len(history),
        len(history_for_prompt),
    )

    def stream_generator():
        """惰性生成答案并在完成后存储消息历史。"""

        full_response = ""
        try:
            answer_generator = rag_handler_instance.get_answer(request.question, history_for_prompt)
            for chunk in answer_generator:
                full_response += chunk
                yield chunk

            # Save assistant response after stream is complete
            db_session = next(get_db())
            assistant_message = DBMessage(chat_id=request.chat_id, role="assistant", content=full_response)
            db_session.add(assistant_message)
            db_session.commit()
            db_session.close()
            logger.info(
                "对话 %d 的回答生成完成，响应长度 %d 字符。",
                request.chat_id,
                len(full_response),
            )

        except Exception as e:
            logger.error(f"Error during RAG answer generation: {e}", exc_info=True)
            yield "抱歉，回答时出现错误。"
    
    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.post("/api/sql_query", response_class=Response)
async def sql_query(request: SQLQueryRequest, current_user: User = Depends(get_current_active_user)):
    """处理 Text-to-SQL 请求"""
    handler = _require_sql_handler()

    logger.info(f"接收到 SQL 查询请求: '{request.question}'")
    db_schema = handler.get_database_schema()
    generated_sql = handler.generate_sql_query(request.question, db_schema)
    return Response(content=generated_sql, media_type="text/plain")

# --- CPE 漏洞查询路由 ---
@app.post("/api/cve_lookup", response_model=List[Dict[str, Any]])
async def cve_lookup(request: CPELookupRequest, current_user: User = Depends(get_current_active_user)):
    """根据 CPE 字符串检索相关漏洞条目。"""

    return cve_handler.search_vulnerabilities_by_cpe(request.cpe)

@app.post("/api/load_nvd_data")
async def load_nvd_data(request: NVDLoadRequest, current_user: User = Depends(get_current_active_user)):
    """从指定目录批量加载 NVD JSON 数据。"""

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
    """根据路径返回打包后的静态页面或入口文件。"""

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
