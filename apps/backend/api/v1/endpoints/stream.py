"""
PRISM Real-Time Streaming WebSocket Router
==========================================
Handles bidirectional real-time audio chunk streaming, Voice Activity Detection (VAD),
partial vs final transcript commits, language detection, code-switch analysis,
and synchronous translation to English.
"""

import asyncio
import base64
import io
import json
import time
import uuid
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from apps.backend.services.session_manager import session_manager
from ml.inference.engine import InferenceEngine
from ml.inference.language_detector import language_detector

router = APIRouter(tags=["Real-time Streaming"])


@router.websocket("/speech/stream")
async def websocket_speech_stream(websocket: WebSocket):
    await websocket.accept()

    session_id = f"PRISM-STREAM-{uuid.uuid4().hex[:8].upper()}"
    engine = InferenceEngine()
    session_manager.create_session(officer_badge="STREAM-OFFICER")

    audio_chunks: List[np.ndarray] = []
    current_segment_chunks: List[np.ndarray] = []
    silence_frames_count = 0
    segment_counter = 1
    total_audio_samples = 0
    is_paused = False

    accumulated_final_transcripts = []
    accumulated_final_translations = []
    detected_language = "te"

    # Initial session handshake
    await websocket.send_json({
        "type": "SESSION_INIT",
        "session_id": session_id,
        "status": "READY",
        "message": "Stream connected to local self-hosted AI engine.",
    })

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                if is_paused:
                    continue
                raw_bytes = data["bytes"]
                chunk = None
                try:
                    chunk, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                    if chunk.ndim > 1:
                        chunk = np.mean(chunk, axis=1)
                    if sr != 16000:
                        chunk = engine.preprocessor.resample(chunk, sr)
                except Exception:
                    pass

                if chunk is None or len(chunk) == 0:
                    try:
                        chunk = np.frombuffer(raw_bytes, dtype=np.float32)
                    except Exception:
                        chunk = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                except Exception:
                    continue

                msg_type = str(msg.get("type", "audio_chunk")).lower()

                if msg_type in ("start_session", "session_started"):
                    detected_language = msg.get("language_hint") or msg.get("language") or "te"
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session_id,
                        "language": detected_language,
                        "status": "started",
                    })
                    continue

                elif msg_type in ("stop_session", "end_session"):
                    await websocket.send_json({
                        "type": "processing_status",
                        "session_id": session_id,
                        "status": "finalizing",
                    })
                    break

                elif msg_type == "pause_session":
                    is_paused = True
                    await websocket.send_json({
                        "type": "processing_status",
                        "session_id": session_id,
                        "status": "paused",
                    })
                    continue

                elif msg_type == "resume_session":
                    is_paused = False
                    await websocket.send_json({
                        "type": "processing_status",
                        "session_id": session_id,
                        "status": "resumed",
                    })
                    continue

                elif msg_type == "audio_chunk":
                    if is_paused:
                        continue
                    b64_audio = msg.get("audio", "")
                    clean_b64 = b64_audio.split(",")[-1] if "," in b64_audio else b64_audio
                    raw_bytes = base64.b64decode(clean_b64)
                    chunk = None
                    try:
                        chunk, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                        if chunk.ndim > 1:
                            chunk = np.mean(chunk, axis=1)
                        if sr != 16000:
                            chunk = engine.preprocessor.resample(chunk, sr)
                    except Exception:
                        pass
                    if chunk is None or len(chunk) == 0:
                        try:
                            chunk = np.frombuffer(raw_bytes, dtype=np.float32)
                        except Exception:
                            chunk = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    continue
            else:
                continue

            if len(chunk) == 0:
                continue

            audio_chunks.append(chunk)
            current_segment_chunks.append(chunk)
            total_audio_samples += len(chunk)

            # Compute short-term RMS energy for VAD
            frame_energy = float(np.sqrt(np.mean(np.square(chunk))))
            is_voice = frame_energy > 0.012

            current_time_sec = round(total_audio_samples / 16000.0, 2)
            segment_start_sec = round((total_audio_samples - sum(len(c) for c in current_segment_chunks)) / 16000.0, 2)

            if is_voice:
                silence_frames_count = 0
                current_buffered_audio = np.concatenate(current_segment_chunks)
                if len(current_buffered_audio) >= 16000 * 0.5:
                    asr_res = engine.asr_provider.transcribe_segment(
                        current_buffered_audio,
                        sample_rate=16000,
                        language_hint=detected_language,
                    )
                    if asr_res.text:
                        seg_id_str = f"seg_{segment_counter:03d}"
                        partial_payload = {
                            "type": "partial_transcript",
                            "session_id": session_id,
                            "segment_id": seg_id_str,
                            "start_time": segment_start_sec,
                            "end_time": current_time_sec,
                            "status": "partial",
                            "language": asr_res.language or detected_language,
                            "transcript": asr_res.text,
                            "translation": "",
                            "confidence": round(asr_res.confidence, 2),
                            "is_partial": True,
                        }
                        # Send both lowercase and legacy keys for compatibility
                        partial_payload["text"] = asr_res.text
                        await websocket.send_json(partial_payload)
            else:
                silence_frames_count += 1
                if silence_frames_count >= 3 and len(current_segment_chunks) > 0:
                    segment_audio = np.concatenate(current_segment_chunks)
                    if len(segment_audio) >= 16000 * 0.4:
                        asr_res = engine.asr_provider.transcribe_segment(
                            segment_audio,
                            sample_rate=16000,
                            language_hint=detected_language,
                        )
                        if asr_res.text:
                            cs_res = engine.code_switch_analyzer.analyze(
                                asr_res.text, language_hint=asr_res.language
                            )
                            lang_det = language_detector.detect_from_text(
                                asr_res.text, language_hint=cs_res.primary_language
                            )
                            trans_res = engine.translation_provider.translate(
                                asr_res.text,
                                source_lang=lang_det.language_code,
                                target_lang="en",
                            )

                            accumulated_final_transcripts.append(asr_res.text)
                            accumulated_final_translations.append(trans_res.translated_text)

                            seg_id_str = f"seg_{segment_counter:03d}"
                            segment_end_sec = current_time_sec

                            # Emit language detection event
                            await websocket.send_json({
                                "type": "language_detected",
                                "session_id": session_id,
                                "language": lang_det.language_code,
                                "language_name": lang_det.language_name,
                                "is_code_switched": cs_res.is_code_switched,
                                "languages": cs_res.languages,
                                "confidence": lang_det.confidence,
                            })

                            # Emit translation event
                            await websocket.send_json({
                                "type": "translation",
                                "session_id": session_id,
                                "segment_id": seg_id_str,
                                "translation": trans_res.translated_text,
                            })

                            # Emit final transcript segment
                            final_payload = {
                                "type": "final_transcript",
                                "session_id": session_id,
                                "segment_id": seg_id_str,
                                "start_time": segment_start_sec,
                                "end_time": segment_end_sec,
                                "status": "final",
                                "language": lang_det.language_code,
                                "transcript": asr_res.text,
                                "translation": trans_res.translated_text,
                                "confidence": round(asr_res.confidence, 2),
                                "is_code_switched": cs_res.is_code_switched,
                                "languages": cs_res.languages,
                                "is_partial": False,
                            }
                            # Compatibility fields
                            final_payload["text"] = asr_res.text
                            await websocket.send_json(final_payload)
                            segment_counter += 1

                    current_segment_chunks = []
                    silence_frames_count = 0

    except WebSocketDisconnect:
        print(f"[PRISM Stream] Session {session_id} disconnected gracefully.")
    except Exception as err:
        print(f"[PRISM Stream Error] {err}")
        try:
            await websocket.send_json({
                "type": "error",
                "session_id": session_id,
                "error": str(err),
            })
        except Exception:
            pass
    finally:
        full_transcript = " ".join(accumulated_final_transcripts).strip()
        full_translation = " ".join(accumulated_final_translations).strip()

        session_manager.update_session(
            session_id,
            {
                "status": "COMPLETED",
                "transcript": full_transcript,
                "english_translation": full_translation,
            },
        )
        try:
            await websocket.send_json({
                "type": "session_completed",
                "session_id": session_id,
                "final_transcript": full_transcript,
                "final_english_translation": full_translation,
            })
            await websocket.close()
        except Exception:
            pass
