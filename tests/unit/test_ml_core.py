"""
PRISM Unit Tests: ML Core Components
====================================
Tests AudioPreprocessor, CodeSwitchAnalyzer, ASR/Translation Providers, and InferenceEngine.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ml.preprocessing.audio import AudioPreprocessor, AudioMetadata
from ml.inference.code_switch import CodeSwitchAnalyzer
from ml.inference.asr_provider import IndicConformerASRProvider
from ml.inference.translation_provider import IndicTrans2Provider
from ml.inference.engine import InferenceEngine


class TestPRISMMlCore(unittest.TestCase):

    def setUp(self):
        self.preprocessor = AudioPreprocessor()
        self.code_switch_analyzer = CodeSwitchAnalyzer()

    def test_audio_preprocessing_pipeline(self):
        # Generate synthetic 1-second 16kHz sine wave audio
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        processed, meta = self.preprocessor.process(audio, original_sr=sr)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(meta.sample_rate, 16000)
        self.assertGreater(meta.duration_sec, 0.5)
        self.assertTrue(meta.is_valid_speech)
        self.assertGreater(meta.snr_db, 0.0)

    def test_code_switch_analyzer_telugu_english(self):
        text = "Na phone railway station daggara theft ayyindi"
        res = self.code_switch_analyzer.analyze(text, language_hint="te")
        self.assertTrue(res.is_code_switched)
        self.assertIn("te", res.languages)
        self.assertIn("en", res.languages)
        self.assertGreater(res.switch_points_count, 0)
        self.assertGreater(len(res.segments), 1)

    def test_code_switch_analyzer_pure_english(self):
        text = "My wallet was stolen at the bus stop yesterday evening"
        res = self.code_switch_analyzer.analyze(text, language_hint="en")
        self.assertEqual(res.primary_language, "en")
        self.assertFalse(res.is_code_switched)

    def test_code_switch_analyzer_native_script(self):
        text = "నా ఫోన్ దొంగిలించబడింది"  # Telugu script
        res = self.code_switch_analyzer.analyze(text)
        self.assertEqual(res.primary_language, "te")

    def test_translation_police_domain(self):
        provider = IndicTrans2Provider()
        res = provider.translate("Na phone railway station daggara theft ayyindi", source_lang="te", target_lang="en")
        self.assertIsInstance(res.translated_text, str)
        self.assertGreater(len(res.translated_text), 5)
        self.assertEqual(res.target_language, "en")

    def test_inference_engine_lifecycle(self):
        engine = InferenceEngine()
        status = engine.get_health_status()
        self.assertIn("status", status)
        self.assertIn("asr", status)
        self.assertIn("translation", status)
        self.assertIn("metrics", status)


if __name__ == "__main__":
    unittest.main()
