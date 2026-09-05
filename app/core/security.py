import asyncio
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import settings


async def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode("utf-8")
    hash_bytes = hashed_password.encode("utf-8")

    def check():
        try:
            return bcrypt.checkpw(password_bytes, hash_bytes)
        except ValueError:
            return False

    return await asyncio.to_thread(check)


async def get_password_hash(password: str) -> str:
    password_bytes = password.encode("utf-8")

    def hash_pw():
        salt = bcrypt.gensalt()
        hashed_bytes = bcrypt.hashpw(password_bytes, salt)
        return hashed_bytes.decode("utf-8")

    return await asyncio.to_thread(hash_pw)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def create_refresh_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def create_storage_token(data: dict, expires_in: int = 1800) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    to_encode.update({"exp": expire, "purpose": "storage"})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def verify_storage_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        if payload.get("purpose") != "storage":
            return None
        return payload
    except Exception:
        return None
