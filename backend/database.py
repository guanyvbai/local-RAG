# backend/database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
from dotenv import load_dotenv
import logging

# 延迟导入，避免循环引用
def get_password_hash(password):
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
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    logger.info(f"成功为应用数据库创建 SQLAlchemy 引擎 (URL: {APP_DATABASE_URL})。")
except Exception as e:
    logger.error(f"创建应用数据库引擎失败: {e}", exc_info=True)
    exit(1)


# --- ORM 模型 (这些表将存在于您的应用数据库中) ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")

class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), default="新的对话", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    chat = relationship("Chat", back_populates="messages")


# --- 数据库初始化和依赖注入 ---

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("应用数据库的表已检查并按需创建。")
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
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()