from fastapi import APIRouter, Depends, status
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import get_current_user
from app.database import get_db
from app.models.users import (
    LoginRequest,
    PasswordResetRequest,
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from app.repositories.users import UserRepository
from app.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def get_auth_service(db: AsyncClient = Depends(get_db)) -> AuthService:
    user_repo = UserRepository(db)
    return AuthService(user_repo)


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate, auth_service: AuthService = Depends(get_auth_service)):
    return await auth_service.register_user(user_data)


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    return await auth_service.login(request.email, request.password)


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(request: PasswordResetRequest, auth_service: AuthService = Depends(get_auth_service)):
    # Note: In a real app, this should involve sending a secure token to the user's email first!
    await auth_service.change_password(request.email, request.new_password)
    return {"message": "Password updated successfully."}


@router.get("/me", response_model=UserResponse)
async def get_my_profile(current_user=Depends(get_current_user)):
    return current_user


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(request: RefreshRequest, auth_service: AuthService = Depends(get_auth_service)):
    return await auth_service.refresh_tokens(request.refresh_token)
