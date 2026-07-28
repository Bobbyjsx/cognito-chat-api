import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from google.cloud.firestore_v1.async_client import AsyncClient

from app.core.config import settings
from app.database import get_db
from app.models.users import UserDB

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

        # Reconstruct the user object from the JWT cache
        return UserDB(
            id=user_id,
            email=payload.get("email"),
            hashed_password=payload.get("hashed_password"),
            tokens_used=payload.get("tokens_used", 0),
            token_limit=payload.get("token_limit", 50000),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )
    except jwt.PyJWTError:
        raise credentials_exception
