"""
PRISM Auth Router
=================
Handles user authentication and profile retrieval.
"""

from fastapi import APIRouter, HTTPException, status
from apps.backend.core.security import authenticate_user, create_access_token, get_current_user
from apps.backend.models.schemas import Token, UserLogin, UserOut
from fastapi import Depends

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user.username, "role": user.role})
    return Token(access_token=token, token_type="bearer", user=user)


@router.get("/me", response_model=UserOut)
async def get_me(current_user: UserOut = Depends(get_current_user)):
    return current_user
