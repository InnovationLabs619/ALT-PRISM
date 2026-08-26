"""
PRISM Pydantic Schemas & DTOs
=============================
Defines request and response data structures for REST endpoints and WebSockets.
"""

import enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- Auth & RBAC Schemas ---
class UserRole(str, enum.Enum):
    ADMIN = "ADMIN"
    OFFICER = "OFFICER"


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    username: str
    full_name: str
    badge_number: str
    department: str
    role: UserRole


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    role: Optional[str] = None
    exp: Optional[int] = None


# --- Speech Schemas ---
class ASRSegmentSchema(BaseModel):
    segment_id: int
    start_sec: float
    end_sec: float
    text: str
    confidence: float
    language: str


class TimingMetricsSchema(BaseModel):
    audio_duration: float
    preprocessing_latency: float
    asr_latency: float
    code_switch_latency: float
    translation_latency: float
    total_latency: float


class AudioQualitySchema(BaseModel):
    snr_db: float = 0.0
    quality_score: float = 0.0
    channels: int = 1
    sample_rate: int = 16000


class SpeechProcessResponse(BaseModel):
    session_id: str
    language: str
    language_confidence: float
    is_code_switched: bool
    languages: List[str]
    transcript: str
    english_translation: str
    segments: List[ASRSegmentSchema]
    timing: TimingMetricsSchema
    models: Dict[str, str]
    audio_quality: AudioQualitySchema
    status: str
    error_message: Optional[str] = None


class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = None
    language_hint: Optional[str] = "te"


class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    confidence: float
    segments: List[ASRSegmentSchema]
    latency_sec: float
    model_name: str


class TranslateRequest(BaseModel):
    text: str
    source_language: str = "te"
    target_language: str = "en"


class TranslateResponse(BaseModel):
    source_text: str
    english_translation: str
    source_language: str
    target_language: str
    confidence: float
    latency_sec: float
    model_name: str


# --- Session Schemas ---
class SessionStatus(str, enum.Enum):
    CREATED = "CREATED"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DELETED = "DELETED"


class SessionSaveRequest(BaseModel):
    transcript: Optional[str] = None
    english_translation: Optional[str] = None
    status: Optional[str] = None
    officer_notes: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    officer_badge: str
    status: SessionStatus
    created_at: str
    updated_at: str
    duration_seconds: float
    primary_language: Optional[str] = None
    is_code_switched: Optional[bool] = None
    transcript: Optional[str] = None
    english_translation: Optional[str] = None
    metadata: Dict[str, Any] = {}


# --- Health & Diagnostics Schemas ---
class ModelStatusSchema(BaseModel):
    status: str
    model: str
    version: str


class AIHealthResponse(BaseModel):
    status: str
    device: str
    vram_allocated_mb: Optional[float] = None
    asr: ModelStatusSchema
    translation: ModelStatusSchema
    metrics: Dict[str, Any]


class SampleAudioItem(BaseModel):
    audio_id: str
    filename: str
    language: str
    language_name: str
    duration_sec: float
    incident_category: str
    expected_transcript: str
    expected_translation: str
    is_code_switched: bool
