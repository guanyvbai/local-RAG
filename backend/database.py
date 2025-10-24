# backend/database.py
"""Database utilities and SQLAlchemy models.

为了方便理解，本文档集中解释了应用数据库初始化流程：

1. 读取 ``APP_DATABASE_URL`` 环境变量并创建 SQLAlchemy 引擎；
2. 定义 ``User``、``Chat``、``Message`` 三张业务表的 ORM 模型；
3. 在应用启动时调用 :func:`init_db` 初始化表结构并预置一个管理员账号；
4. 通过 :func:`get_db` 提供 FastAPI 依赖，按请求生命周期管理数据库会话。
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
from dotenv import load_dotenv
import logging

# 延迟导入，避免循环引用
def get_password_hash(password):
    """Proxy to :func:`auth.get_password_hash` to avoid circular imports."""

    from auth import get_password_hash as auth_hash
    return auth_hash(password)


load_dotenv()
logger = logging.getLogger(__name__)

# 【核心修改】使用新的环境变量 APP_DATABASE_URL
APP_DATABASE_URL = os.getenv("APP_DATABASE_URL")

if not APP_DATABASE_URL:
    logger.critical("错误：环境变量 APP_DATABASE_URL 未设置！应用无法启动。")
    exit(1)

# --- 为应用数据库设置 SQLAlchemy 引擎和会话 ---
try:
    # 【核心修改】确保引擎连接到 APP_DATABASE_URL
    engine = create_engine(APP_DATABASE_URL, pool_pre_ping=True)
    # SessionLocal 作为工厂函数为每个请求提供独立的会话。
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    # declarative_base 生成的 Base 类是所有 ORM 模型的基类。
    Base = declarative_base()
    logger.info(f"成功为应用数据库创建 SQLAlchemy 引擎 (URL: {APP_DATABASE_URL})。")
except Exception as e:
    logger.error(f"创建应用数据库引擎失败: {e}", exc_info=True)
    exit(1)


# --- ORM 模型 (这些表将存在于您的应用数据库中) ---


class User(Base):
    """Represent an application user with login credentials and chat sessions."""

    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")


class Chat(Base):
    """Store a single conversation thread belonging to a :class:`User`."""

    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), default="新的对话", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")


class Message(Base):
    """Persist a single message belonging to a :class:`Chat`."""

    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    chat = relationship("Chat", back_populates="messages")


# --- 数据库初始化和依赖注入 ---


def init_db() -> None:
    """Create tables (if needed) and ensure a default ``admin`` user exists."""

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("应用数据库的表已检查并按需创建。")
        # 使用 get_db() 生成的上下文来复用统一的会话管理逻辑。
        db = next(get_db())
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            hashed_password = get_password_hash("admin")
            db_user = User(username="admin", hashed_password=hashed_password)
            db.add(db_user)
            db.commit()
            logger.info("默认 admin 用户已在应用数据库中创建。")
        db.close()
    except Exception as e:
        logger.error(f"应用数据库初始化时出错: {e}", exc_info=True)


def get_db():
    """Yield a database session and ensure it is closed after use."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()