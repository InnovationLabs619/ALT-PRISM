"""
PRISM Security & RBAC Module
============================
Provides JWT authentication and Role-Based Access Control (ADMIN, OFFICER)
for internal police department operations.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
import hashlib
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from apps.backend.core.config import settings
from apps.backend.models.schemas import UserOut, UserRole

security_scheme = HTTPBearer(auto_error=False)

# Internal authorized department accounts
INTERNAL_USERS_DB: Dict[str, Dict] = {
    "officer_104": {
        "username": "officer_104",
        "full_name": "Inspector K. Rajesh",
        "badge_number": "AP-POL-10492",
        "department": "Law & Order - Cyberabad",
        "role": UserRole.OFFICER,
        "password_hash": hashlib.sha256("PrismOfficer@2026".encode()).hexdigest(),
    },
    "admin_hq": {
        "username": "admin_hq",
        "full_name": "DCP S. Venkat Raman",
        "badge_number": "AP-POL-00018",
        "department": "State Intelligence & AI Ops",
        "role": UserRole.ADMIN,
        "password_hash": hashlib.sha256("PrismAdmin@2026".encode()).hexdigest(),
    },
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    candidate_hash = hashlib.sha256(plain_password.encode()).hexdigest()
    return candidate_hash == hashed_password


def authenticate_user(username: str, password: str) -> Optional[UserOut]:
    user = INTERNAL_USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return UserOut(
        username=user["username"],
        full_name=user["full_name"],
        badge_number=user["badge_number"],
        department=user["department"],
        role=user["role"],
    )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> UserOut:
    """Authenticates the incoming bearer token or allows local demo access if header missing."""
    if not credentials:
        # Default internal development officer session if no token provided
        return UserOut(
            username="officer_104",
            full_name="Inspector K. Rajesh",
            badge_number="AP-POL-10492",
            department="Law & Order - Cyberabad",
            role=UserRole.OFFICER,
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    user = INTERNAL_USERS_DB.get(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return UserOut(
        username=user["username"],
        full_name=user["full_name"],
        badge_number=user["badge_number"],
        department=user["department"],
        role=user["role"],
    )


def require_admin(current_user: UserOut = Depends(get_current_user)) -> UserOut:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this operation",
        )
    return current_user
