# backend/auth.py (已修正最终拼写错误)
"""Authentication helpers for the FastAPI backend.

该模块集中处理密码哈希、JWT 生成/校验以及 OAuth2 依赖的构建。由于
这些函数会在多个路由中被复用，我们通过在这里编写详细注释来说明每个
步骤的目的，方便后续维护人员迅速理解安全相关逻辑。
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# --- 配置 ---
# SECRET_KEY 用于对 JWT 进行签名，默认值仅用于本地开发环境。
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-for-jwt-!@#$%^")
# HS256 是对称加密算法，编码端与解码端共享 SECRET_KEY。
ALGORITHM = "HS256"
# 访问令牌默认有效期 7 天。为了便于阅读，以分钟为单位进行配置。
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7天

# --- 密码处理 ---
# Passlib 会为我们处理盐值以及安全的哈希算法选择（当前配置为 bcrypt）。
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- OAuth2 & JWT ---
# FastAPI 的 OAuth2PasswordBearer 依赖会自动从请求头中解析 "Authorization"
# bearer token，并交由后续的解码函数验证。
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return ``True`` if *plain_password* matches the stored hash.

    FastAPI 的用户登录流程会先从数据库中读取 ``hashed_password``，再用
    用户提交的明文密码进行校验。借助 Passlib 可以避免手动处理盐值和
    哈希算法细节，从而降低安全风险。
    """

    # 【修正】: 将 pwd_content.verify 改为 pwd_context.verify
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash *password* using the configured :class:`CryptContext`.

    在注册用户或初始化默认管理员账号时调用。统一由该函数生成的哈希值
    存入数据库，避免在代码中出现明文密码。
    """

    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT containing *data* and an expiration timestamp.

    ``expires_delta`` 允许调用者覆写默认有效期，例如在测试时缩短有效期。
    最终的 payload 会包含所有原始字段以及 ``exp``（过期时间）。
    """

    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Decode *token* and return the payload when it is valid.

    若令牌已过期或签名无效，函数返回 ``None``。调用者可以据此决定是否
    抛出 401 错误。为了方便后续的业务逻辑，我们在这里额外检查 ``sub``
    字段（用户名）是否存在。
    """

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            return None
        return payload
    except JWTError:
        return None