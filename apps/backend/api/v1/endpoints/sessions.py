"""
PRISM Sessions Router
=====================
Provides retrieval, listing, and deletion of temporary speech processing sessions.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from apps.backend.core.security import get_current_user
from apps.backend.models.schemas import SessionResponse, UserOut
from apps.backend.services.session_manager import session_manager

router = APIRouter(prefix="/sessions", tags=["Session Management"])


@router.get("", response_model=List[SessionResponse])
async def list_active_sessions(current_user: UserOut = Depends(get_current_user)):
    return session_manager.list_sessions()


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session_by_id(
    session_id: str, current_user: UserOut = Depends(get_current_user)
):
    sess = session_manager.get_session(session_id)
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    return sess


@router.delete("/{session_id}")
async def delete_session_by_id(
    session_id: str, current_user: UserOut = Depends(get_current_user)
):
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    return {"message": f"Session {session_id} deleted successfully."}
