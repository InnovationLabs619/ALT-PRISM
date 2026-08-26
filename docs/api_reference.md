# PRISM REST & WebSocket API Reference
======================================

Base API Prefix: `/api/v1`

---

## 1. Authentication Endpoints

### `POST /api/v1/auth/login`
Authenticates department personnel and issues JWT bearer tokens.

**Request Body:**
```json
{
  "username": "officer_104",
  "password": "PrismOfficer@2026"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1Ni...",
  "token_type": "bearer",
  "user": {
    "username": "officer_104",
    "full_name": "Inspector K. Rajesh",
    "badge_number": "AP-POL-10492",
    "department": "Law & Order - Cyberabad",
    "role": "OFFICER"
  }
}
```

---

## 2. Speech Processing Endpoints

### `POST /api/v1/speech/process`
Unified multi-stage speech inference endpoint accepting multipart audio files or Base64 payloads.

**Parameters (Multipart / Form-Data):**
- `file`: Audio file (`.wav`, `.mp3`, `.m4a`, `.webm`) [Optional if `audio_base64` passed]
- `audio_base64`: Base64 encoded audio string [Optional if `file` passed]
- `language_hint`: e.g., `"te"`, `"hi"`, `"ta"`, `"kn"`, `"mr"`, `"bn"`, `"ml"`, `"en"` [Optional]
- `target_language`: Default `"en"`
- `session_id`: Optional custom session identifier

**Response (200 OK):**
```json
{
  "session_id": "PRISM-SES-8F29A10C",
  "language": "te",
  "language_confidence": 0.94,
  "is_code_switched": true,
  "languages": ["te", "en"],
  "transcript": "Na phone ninna evening railway station daggara theft ayyindi.",
  "english_translation": "My phone was stolen near the railway station yesterday evening.",
  "segments": [
    {
      "segment_id": 0,
      "start_sec": 0.0,
      "end_sec": 3.8,
      "text": "Na phone ninna evening railway station daggara theft ayyindi.",
      "confidence": 0.96,
      "language": "te"
    }
  ],
  "timing": {
    "audio_duration": 3.8,
    "preprocessing_latency": 0.0084,
    "asr_latency": 0.421,
    "code_switch_latency": 0.0005,
    "translation_latency": 0.892,
    "total_latency": 1.322
  },
  "models": {
    "asr": "openai/whisper-tiny",
    "asr_version": "1.0",
    "translation": "facebook/nllb-200-distilled-600M",
    "translation_version": "1.0",
    "pipeline_version": "prism-mvp-1.0"
  },
  "audio_quality": {
    "snr_db": 22.4,
    "quality_score": 0.92,
    "channels": 1,
    "sample_rate": 16000
  },
  "status": "success"
}
```

---

## 3. Real-Time Streaming WebSocket

### `WebSocket /api/v1/speech/stream`
Bidirectional WebSocket for live audio capture and streaming inference.

**Client Message (Handshake):**
```json
{
  "type": "START_SESSION",
  "language_hint": "te"
}
```

**Client Message (Audio Chunk):**
```json
{
  "type": "AUDIO_CHUNK",
  "audio": "<base64_encoded_webm_or_pcm>"
}
```

**Server Message (Partial Transcription):**
```json
{
  "type": "PARTIAL_TRANSCRIPT",
  "session_id": "PRISM-STREAM-4D2A",
  "segment_id": 0,
  "text": "Na phone ninna evening",
  "language": "te",
  "is_partial": true
}
```

**Server Message (Final Segment with Translation):**
```json
{
  "type": "FINAL_SEGMENT",
  "session_id": "PRISM-STREAM-4D2A",
  "segment_id": 0,
  "text": "Na phone ninna evening railway station daggara theft ayyindi.",
  "translation": "My phone was stolen near the railway station yesterday evening.",
  "language": "te",
  "is_code_switched": true,
  "confidence": 0.95,
  "is_partial": false
}
```

---

## 4. Diagnostics & Health

- `GET /api/v1/health` -> System health status.
- `GET /api/v1/health/ai` -> Model status, device allocation, VRAM usage, inference count.
- `GET /api/v1/metrics` -> Performance and throughput metrics.
- `GET /api/v1/speech/samples` -> Available domain golden test samples.
- `GET /api/v1/sessions` -> List recent investigation sessions.
