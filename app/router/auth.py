from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import get_current_user
from app.database import get_db
from app.models.users import (
    PasswordResetRequest,
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
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
):
    return await auth_service.login(form_data.username, form_data.password)


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(request: PasswordResetRequest, auth_service: AuthService = Depends(get_auth_service)):
    # Note: In a real app, this should involve sending a secure token to the user's email first!
    await auth_service.change_password(request.email, request.new_password)
    return {"message": "Password updated successfully."}


@router.get("/me", response_model=UserResponse)
async def get_my_profile(current_user=Depends(get_current_user)):
    return current_user
