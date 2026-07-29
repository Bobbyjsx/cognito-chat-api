import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from google.cloud.firestore_v1.async_client import AsyncClient

from app.core.config import settings
from app.database import get_db
from app.models.users import UserDB
from app.repositories.users import UserRepository

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncClient = Depends(get_db)) -> UserDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        if user_id is None or token_type == "refresh":
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    # Always fetch fresh from Firestore so tokens_used reflects live state
    user = await UserRepository(db).get_by_id(user_id)
    if user is None:
        raise credentials_exception
    return user
