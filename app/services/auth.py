from fastapi import HTTPException, status

from app.core.security import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
)
from app.models.users import TokenResponse, UserCreate, UserDB
from app.repositories.users import UserRepository


class AuthService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    async def register_user(self, user_data: UserCreate) -> UserDB:
        existing_user = await self.user_repo.get_by_email(user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_pw = get_password_hash(user_data.password)
        new_user = UserDB(email=user_data.email, hashed_password=hashed_pw)
        return await self.user_repo.create(new_user)

    async def authenticate_user(self, email: str, password: str) -> UserDB | None:
        user = await self.user_repo.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            return None
        return user

    async def login(self, email: str, password: str) -> TokenResponse:
        user = await self.authenticate_user(email, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": str(user.id)})

        return TokenResponse(access_token=access_token, refresh_token=refresh_token)

    async def change_password(self, email: str, new_password: str) -> None:
        user = await self.user_repo.get_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        hashed_pw = get_password_hash(new_password)
        await self.user_repo.update_password(user.id, hashed_pw)
