"""
PRISM Speech Pipeline Automated Test Suite
==========================================
Verifies end-to-end self-hosted pipeline components, WebSocket streaming,
Language Detection, ASR, NMT, Code-Switching, Session Storage, and Document Exports.
"""

import io
import json
import os
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from apps.backend.main import app
from ml.inference.engine import InferenceEngine
from ml.inference.language_detector import LanguageDetector
from ml.inference.code_switch import CodeSwitchAnalyzer
from ml.inference.translation_provider import IndicTrans2Provider
from apps.backend.services.session_manager import session_manager

client = TestClient(app)


# --- 1. Language Detector Unit Tests ---
def test_language_detector_supported_languages():
    detector = LanguageDetector()
    
    # Test Telugu native script
    res_te = detector.detect_from_text("నా పేరు రమేష్ కుమార్, నా ఫోన్ దొంగిలించబడింది")
    assert res_te.language_code == "te"
    assert res_te.language_name == "Telugu"
    assert res_te.confidence >= 0.70

    # Test Hindi native script
    res_hi = detector.detect_from_text("मेरा नाम रमेश है और मेरी गाड़ी चोरी हो गई")
    assert res_hi.language_code == "hi"
    assert res_hi.language_name == "Hindi"

    # Test Romanized Indic detection
    res_rom_te = detector.detect_from_text("naa phone railway station daggara theft ayyindi")
    assert res_rom_te.language_code == "te"

    # Test English
    res_en = detector.detect_from_text("An unknown person called me pretending to be a bank manager")
    assert res_en.language_code == "en"

    # Test unknown / low confidence fallback (does not invent a language)
    res_unk = detector.detect_from_text("xyzq1234!!")
    assert res_unk.language_code == "unknown"


# --- 2. Code-Switch Analyzer Unit Tests ---
def test_code_switch_analyzer():
    analyzer = CodeSwitchAnalyzer()
    
    res = analyzer.analyze("na phone railway station daggara theft ayyindi", language_hint="te")
    assert res.is_code_switched is True
    assert "te" in res.languages
    assert "en" in res.languages
    assert res.code_switch_ratio > 0.0


# --- 3. Translation Provider Unit Tests ---
def test_translation_provider():
    provider = IndicTrans2Provider()
    
    res = provider.translate(
        "na phone railway station daggara theft ayyindi",
        source_language="te",
        target_language="en",
    )
    assert res.translated_text is not None
    assert len(res.translated_text) > 0
    assert res.target_language == "en"


# --- 4. ASR Provider Unit Tests ---
def test_asr_provider_transcribe_segment():
    engine = InferenceEngine()
    # 1 second of 440Hz sine wave audio
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sine_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    res = engine.asr_provider.transcribe_segment(sine_audio, sample_rate=sample_rate, language_hint="te")
    assert res.model_name is not None
    assert hasattr(res, "latency_sec")


# --- 5. Speech Upload Endpoint Tests ---
def test_speech_upload_endpoint():
    # Generate 1s 16kHz mono WAV buffer
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sine_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    wav_io = io.BytesIO()
    sf.write(wav_io, sine_audio, sample_rate, format="WAV")
    wav_io.seek(0)

    response = client.post(
        "/api/v1/speech/upload",
        files={"file": ("test_audio.wav", wav_io, "audio/wav")},
        data={"language_hint": "te"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "transcript" in data
    assert "english_translation" in data
    assert "timing" in data


# --- 6. Standalone Translation Endpoint Test ---
def test_standalone_translate_endpoint():
    response = client.post(
        "/api/v1/speech/translate",
        json={
            "text": "నా ఫోన్ దొంగిలించబడింది",
            "source_language": "te",
            "target_language": "en",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source_text"] == "నా ఫోన్ దొంగిలించబడింది"
    assert len(data["english_translation"]) > 0


# --- 7. Session Save & Document Export Tests ---
def test_session_save_and_exports():
    # Create session
    session = session_manager.create_session(officer_badge="TEST-OFFICER-99")
    session_id = session.session_id

    # Populate session data
    session_manager.update_session(
        session_id,
        {
            "transcript": "Naa peru Ramesh. Phone theft ayyindi.",
            "english_translation": "My name is Ramesh. Phone was stolen.",
            "primary_language": "te",
            "duration_seconds": 3.5,
        },
    )

    # Save updates
    save_resp = client.post(
        f"/api/v1/speech/session/{session_id}/save",
        json={
            "transcript": "Naa peru Ramesh Kumar. Phone theft ayyindi.",
            "english_translation": "My name is Ramesh Kumar. Phone was stolen.",
        },
    )
    assert save_resp.status_code == 200
    saved_data = save_resp.json()
    assert saved_data["transcript"] == "Naa peru Ramesh Kumar. Phone theft ayyindi."

    # Export TXT
    txt_resp = client.get(f"/api/v1/speech/session/{session_id}/export/txt")
    assert txt_resp.status_code == 200
    assert "PRISM POLICE INVESTIGATION STATEMENT TRANSCRIPT" in txt_resp.text
    assert session_id in txt_resp.text

    # Export DOCX
    docx_resp = client.get(f"/api/v1/speech/session/{session_id}/export/docx")
    assert docx_resp.status_code == 200
    assert docx_resp.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# --- 8. AI Health & Observability Endpoint Test ---
def test_ai_health_endpoint():
    response = client.get("/api/v1/health/ai")
    assert response.status_code == 200
    data = response.json()
    assert data["status"].upper() == "READY"
    assert "asr" in data
    assert "translation" in data
    assert "metrics" in data


# --- 9. WebSocket Stream Integration Test ---
def test_websocket_stream_lifecycle():
    with client.websocket_connect("/api/v1/speech/stream") as websocket:
        init_data = websocket.receive_json()
        assert init_data["type"] == "SESSION_INIT"

        websocket.send_json({"type": "start_session", "language_hint": "te"})
        start_data = websocket.receive_json()
        assert start_data["type"] == "session_started"

        # Send raw 16kHz PCM audio chunk (0.5s)
        sample_rate = 16000
        pcm_chunk = (np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)) * 32767).astype(np.int16).tobytes()
        websocket.send_bytes(pcm_chunk)

        # Stop session
        websocket.send_json({"type": "stop_session"})
        
        received_types = []
        for _ in range(5):
            try:
                msg = websocket.receive_json()
                received_types.append(msg.get("type"))
                if msg.get("type") in ("session_completed", "processing_status"):
                    break
            except Exception:
                break

        assert any(t in ("session_completed", "processing_status", "partial_transcript", "final_transcript") for t in received_types)

