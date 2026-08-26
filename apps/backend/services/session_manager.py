"""
PRISM Speech Session Manager
============================
Tracks ephemeral speech processing sessions and provides lifecycle cleanup.
"""

import time
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timezone

from apps.backend.models.schemas import SessionResponse, SessionStatus


class SessionManager:
    """Manages police speech processing sessions in-memory for MVP."""

    _instance: Optional["SessionManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance.sessions: Dict[str, Dict] = {}
        return cls._instance

    def create_session(self, officer_badge: str = "AP-POL-10492") -> SessionResponse:
        session_id = f"PRISM-SES-{uuid.uuid4().hex[:8].upper()}"
        now_str = datetime.now(timezone.utc).isoformat()

        session_data = {
            "session_id": session_id,
            "officer_badge": officer_badge,
            "status": SessionStatus.CREATED,
            "created_at": now_str,
            "updated_at": now_str,
            "duration_seconds": 0.0,
            "primary_language": None,
            "is_code_switched": None,
            "transcript": None,
            "english_translation": None,
            "metadata": {},
        }
        self.sessions[session_id] = session_data
        return SessionResponse(**session_data)

    def update_session(self, session_id: str, updates: Dict) -> Optional[SessionResponse]:
        if session_id not in self.sessions:
            return None
        s = self.sessions[session_id]
        s.update(updates)
        s["updated_at"] = datetime.now(timezone.utc).isoformat()
        return SessionResponse(**s)

    def get_session(self, session_id: str) -> Optional[SessionResponse]:
        s = self.sessions.get(session_id)
        if not s:
            return None
        return SessionResponse(**s)

    def list_sessions(self, limit: int = 50) -> List[SessionResponse]:
        # Return recent sessions
        items = list(self.sessions.values())[-limit:]
        return [SessionResponse(**item) for item in reversed(items)]

    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


session_manager = SessionManager()
