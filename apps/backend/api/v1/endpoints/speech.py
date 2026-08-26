"""
PRISM Speech REST Endpoints
===========================
Provides speech audio upload/processing, standalone ASR and translation,
session detail retrieval/updates, and official Police Statement document exports (.txt, .docx).
"""

import base64
import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from apps.backend.core.config import settings
from apps.backend.core.security import get_current_user
from apps.backend.models.schemas import (
    SampleAudioItem,
    SessionResponse,
    SessionSaveRequest,
    SpeechProcessResponse,
    TranscribeRequest,
    TranscribeResponse,
    TranslateRequest,
    TranslateResponse,
    UserOut,
)
from apps.backend.services.session_manager import session_manager
from ml.inference.engine import InferenceEngine

router = APIRouter(prefix="/speech", tags=["Speech Processing"])


@router.post("/upload", response_model=SpeechProcessResponse)
@router.post("/process", response_model=SpeechProcessResponse)
async def process_speech_audio(
    file: Optional[UploadFile] = File(None),
    audio_base64: Optional[str] = Form(None),
    language_hint: Optional[str] = Form(None),
    target_language: str = Form("en"),
    session_id: Optional[str] = Form(None),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Unified Speech Audio Upload & Processing Pipeline:
    Audio -> Normalization -> VAD -> Indic ASR -> Code-Switch Analysis -> Translation -> Document Setup.
    Calculates Real-Time Factor (RTF = processing_time / audio_duration).
    """
    if not file and not audio_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either an audio file or audio_base64 payload must be provided.",
        )

    active_session_id = session_id or f"PRISM-SES-{uuid.uuid4().hex[:8].upper()}"
    session_manager.create_session(officer_badge=current_user.badge_number)
    session_manager.update_session(active_session_id, {"status": "PROCESSING"})

    audio_bytes = None
    if file:
        audio_bytes = await file.read()
    elif audio_base64:
        clean_b64 = audio_base64.split(",")[-1] if "," in audio_base64 else audio_base64
        audio_bytes = base64.b64decode(clean_b64)

    engine = InferenceEngine()
    result = engine.process_speech(
        audio_input=audio_bytes,
        session_id=active_session_id,
        language_hint=language_hint,
        target_language=target_language,
    )

    # Compute Real-Time Factor (RTF)
    rtf = round(result.timing.total_latency / max(0.1, result.timing.audio_duration), 3)

    session_manager.update_session(
        active_session_id,
        {
            "status": "COMPLETED" if result.status == "success" else "FAILED",
            "duration_seconds": result.timing.audio_duration,
            "primary_language": result.language,
            "is_code_switched": result.is_code_switched,
            "transcript": result.transcript,
            "english_translation": result.english_translation,
            "metadata": {**result.to_dict(), "rtf": rtf},
        },
    )

    return SpeechProcessResponse(**result.to_dict())


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_only(
    request: TranscribeRequest,
    current_user: UserOut = Depends(get_current_user),
):
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="Missing audio_base64")

    clean_b64 = request.audio_base64.split(",")[-1] if "," in request.audio_base64 else request.audio_base64
    audio_bytes = base64.b64decode(clean_b64)

    engine = InferenceEngine()
    clean_audio, meta = engine.preprocessor.process(audio_bytes)
    res = engine.asr_provider.transcribe(
        clean_audio, sample_rate=meta.sample_rate, language_hint=request.language_hint
    )

    return TranscribeResponse(
        transcript=res.text,
        language=res.language,
        confidence=res.confidence,
        segments=[s.to_dict() if hasattr(s, "to_dict") else s.__dict__ for s in res.segments],
        latency_sec=res.latency_sec,
        model_name=res.model_name,
    )


@router.post("/translate", response_model=TranslateResponse)
async def translate_text_only(
    request: TranslateRequest,
    current_user: UserOut = Depends(get_current_user),
):
    engine = InferenceEngine()
    res = engine.translation_provider.translate(
        text=request.text,
        source_lang=request.source_language,
        target_lang=request.target_language,
    )
    return TranslateResponse(
        source_text=res.source_text,
        english_translation=res.translated_text,
        source_language=res.source_language,
        target_language=res.target_language,
        confidence=res.confidence,
        latency_sec=res.latency_sec,
        model_name=res.model_name,
    )


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session_details(
    session_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    """Retrieves session record details by session ID."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return session


@router.post("/session/{session_id}/save", response_model=SessionResponse)
async def save_session_updates(
    session_id: str,
    payload: SessionSaveRequest,
    current_user: UserOut = Depends(get_current_user),
):
    """
    Saves officer manual edits to the transcript, translation, or session notes.
    Never silently overwrites officer edits.
    """
    updates = {}
    if payload.transcript is not None:
        updates["transcript"] = payload.transcript
    if payload.english_translation is not None:
        updates["english_translation"] = payload.english_translation
    if payload.status is not None:
        updates["status"] = payload.status
    if payload.officer_notes is not None:
        updates["officer_notes"] = payload.officer_notes

    updated_session = session_manager.update_session(session_id, updates)
    if not updated_session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return updated_session


@router.get("/session/{session_id}/export/txt")
async def export_transcript_txt(
    session_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    """Exports session transcript as formatted plain text (.txt)."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    dt_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    txt_content = f"""================================================================================
PRISM POLICE INVESTIGATION STATEMENT TRANSCRIPT
================================================================================
Session ID        : {session.session_id}
Officer Badge     : {session.officer_badge}
Recorded Date     : {session.created_at}
Export Date       : {dt_now}
Primary Language  : {session.primary_language or 'Auto-Detected'}
Audio Duration    : {session.duration_seconds:.2f} seconds
Classification    : RESTRICTED LAW ENFORCEMENT RECORD

================================================================================
1. ORIGINAL STATEMENT TRANSCRIPT
================================================================================
{session.transcript or 'No statement recorded.'}

================================================================================
2. ENGLISH TRANSLATION
================================================================================
{session.english_translation or 'No translation recorded.'}

================================================================================
3. SYSTEM DIAGNOSTICS & TELEMETRY
================================================================================
ASR Model         : IndicConformer Multilingual (Self-Hosted)
Translation Model : IndicTrans2 / NLLB-200 (Self-Hosted)
AI Architecture   : 100% Local Air-Gapped Inference Engine
================================================================================
"""
    filename = f"PRISM_Statement_{session.session_id}.txt"
    return Response(
        content=txt_content,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/session/{session_id}/export/docx")
async def export_transcript_docx(
    session_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    """Exports session transcript as an official Word document (.docx)."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    doc = docx.Document()

    # Document Header
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("PRISM — POLICE INVESTIGATION STATEMENT TRANSCRIPT")
    r_title.bold = True
    r_title.font.size = Pt(16)
    r_title.font.color.rgb = RGBColor(15, 23, 42)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Subtitle / Notice
    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run("OFFICIAL LAW ENFORCEMENT EVIDENTIARY DOCUMENT (100% LOCAL AI)")
    r_sub.font.size = Pt(9)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(100, 116, 139)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Metadata Table
    table = doc.add_table(rows=5, cols=2)
    table.style = 'Table Grid'
    metadata_pairs = [
        ("Session Reference ID", session.session_id),
        ("Investigating Officer Badge", session.officer_badge),
        ("Recording Timestamp", session.created_at),
        ("Primary Spoken Language", session.primary_language or 'Auto-Detected'),
        ("Audio Duration", f"{session.duration_seconds:.2f} Seconds"),
    ]
    for idx, (label, val) in enumerate(metadata_pairs):
        cell_lbl, cell_val = table.rows[idx].cells
        cell_lbl.paragraphs[0].add_run(label).bold = True
        cell_val.paragraphs[0].add_run(str(val))

    doc.add_heading("1. Original Statement Transcript", level=1)
    p_orig = doc.add_paragraph()
    p_orig.add_run(session.transcript or "No statement recorded.")

    doc.add_heading("2. Official English Translation", level=1)
    p_trans = doc.add_paragraph()
    p_trans.add_run(session.english_translation or "No translation recorded.")

    doc.add_heading("3. AI Pipeline & Security Metadata", level=1)
    doc.add_paragraph("ASR Engine: IndicConformer / IndicWhisper Multilingual (Local)")
    doc.add_paragraph("Translation Model: IndicTrans2 / NLLB-200 (Local)")
    doc.add_paragraph("Cloud Data Transfer: ZERO (100% Air-Gapped Offline Inference)")

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    filename = f"PRISM_Statement_{session.session_id}.docx"
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/samples", response_model=List[SampleAudioItem])
async def list_sample_audios():
    """Returns available golden test samples for interactive verification."""
    manifest_file = settings.BASE_DIR / "tests" / "golden" / "golden_test_set.json"
    if not manifest_file.exists():
        return []

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    lang_names = {
        "te": "Telugu (తెలుగు)",
        "hi": "Hindi (हिन्दी)",
        "ta": "Tamil (தமிழ்)",
        "kn": "Kannada (ಕನ್ನಡ)",
        "mr": "Marathi (मराठी)",
        "bn": "Bengali (বাংলা)",
        "ml": "Malayalam (മലയാളം)",
        "en": "English",
    }

    result = []
    for item in data:
        result.append(
            SampleAudioItem(
                audio_id=item["audio_id"],
                filename=item["filename"],
                language=item["language"],
                language_name=lang_names.get(item["language"], item["language"].upper()),
                duration_sec=item["duration"],
                incident_category=item["incident_category"],
                expected_transcript=item["transcript"],
                expected_translation=item["english_translation"],
                is_code_switched=item.get("is_code_switched", False),
            )
        )
    return result


@router.get("/sample/{filename}")
async def get_sample_audio_file(filename: str):
    """Serves sample WAV audio for streaming and testing."""
    file_path = settings.SAMPLES_DIR / filename
    if not file_path.exists():
        file_path = settings.BASE_DIR / "tests" / "golden" / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample audio {filename} not found.")

    return FileResponse(file_path, media_type="audio/wav", filename=filename)
