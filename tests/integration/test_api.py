"""
PRISM Backend Integration Tests
================================
Tests FastAPI REST endpoints, auth, and speech processing using TestClient.
"""

import io
import os
import sys
import unittest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from apps.backend.main import app


class TestPRISMBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoints(self):
        # Health check
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

        # AI Health check
        res_ai = self.client.get("/api/v1/health/ai")
        self.assertEqual(res_ai.status_code, 200)
        ai_data = res_ai.json()
        self.assertIn("asr", ai_data)
        self.assertIn("translation", ai_data)
        self.assertIn("device", ai_data)

    def test_02_auth_login(self):
        payload = {
            "username": "officer_104",
            "password": "PrismOfficer@2026"
        }
        res = self.client.post("/api/v1/auth/login", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["user"]["badge_number"], "AP-POL-10492")

    def test_03_samples_endpoint(self):
        res = self.client.get("/api/v1/speech/samples")
        self.assertEqual(res.status_code, 200)
        samples = res.json()
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)
        self.assertEqual(samples[0]["language"], "te")

    def test_04_translate_endpoint(self):
        payload = {
            "text": "Na phone railway station daggara theft ayyindi",
            "source_language": "te",
            "target_language": "en"
        }
        res = self.client.post("/api/v1/speech/translate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("english_translation", data)
        self.assertGreater(len(data["english_translation"]), 5)

    def test_05_process_speech_unified_file(self):
        # Synthesize a 2-second 16kHz WAV in memory
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        wav_io = io.BytesIO()
        sf.write(wav_io, audio, sr, format="WAV", subtype="PCM_16")
        wav_io.seek(0)

        response = self.client.post(
            "/api/v1/speech/process",
            files={"file": ("test_audio.wav", wav_io.getvalue(), "audio/wav")},
            data={"language_hint": "te", "target_language": "en"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("session_id", data)
        self.assertIn("timing", data)
        self.assertIn("models", data)
        self.assertIn("audio_quality", data)

    def test_06_sessions_endpoints(self):
        res = self.client.get("/api/v1/sessions")
        self.assertEqual(res.status_code, 200)
        sessions = res.json()
        self.assertIsInstance(sessions, list)


if __name__ == "__main__":
    unittest.main()
