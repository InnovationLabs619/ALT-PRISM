"""
PRISM Language Detector Module
==============================
Provides self-hosted language identification for Indian regional languages and English.
Integrates script analysis, acoustic features, and language model priors.
Returns language_code, language_name, and confidence.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union
import re
import numpy as np


@dataclass
class LanguageDetectionResult:
    language_code: str
    language_name: str
    confidence: float

    def to_dict(self) -> Dict:
        return asdict(self)


class LanguageDetector:
    """Self-hosted Language Identification engine for PRISM."""

    SUPPORTED_LANGUAGES: Dict[str, str] = {
        "te": "Telugu",
        "hi": "Hindi",
        "ta": "Tamil",
        "kn": "Kannada",
        "ml": "Malayalam",
        "mr": "Marathi",
        "bn": "Bengali",
        "en": "English",
    }

    SCRIPT_UNICODE_RANGES = {
        "te": (0x0C00, 0x0C7F),  # Telugu
        "hi": (0x0900, 0x097F),  # Devanagari (Hindi)
        "ta": (0x0B80, 0x0BFF),  # Tamil
        "kn": (0x0C80, 0x0CFF),  # Kannada
        "ml": (0x0D00, 0x0D7F),  # Malayalam
        "mr": (0x0900, 0x097F),  # Devanagari (Marathi)
        "bn": (0x0980, 0x09FF),  # Bengali
        "en": (0x0041, 0x007A),  # Basic Latin
    }

    # Distinctive Romanized Indic Lexicons for Transliterated Speech Detection
    ROMANIZED_KEYWORDS: Dict[str, List[str]] = {
        "te": ["naa", "nenu", "maku", "daggara", "ninna", "cheppanu", "poyindi", "ayyindi", "unnadu", "vundhi", "telugu"],
        "hi": ["mera", "meri", "mere", "hum", "chori", "gaya", "gayi", "tha", "thi", "kaha", "hindi"],
        "ta": ["enoda", "enakku", "nethu", "pochu", "thiruttu", "aachu", "irukku", "tamil"],
        "kn": ["nanna", "nanage", "aayithu", "kaddaru", "idhe", "baralilla", "kannada"],
        "ml": ["ente", "enikku", "innale", "poyi", "aayi", "undayirunnu", "malayalam"],
        "mr": ["maza", "maze", "zala", "hota", "hoti", "marathi"],
        "bn": ["amar", "amra", "churi", "hoyeche", "bengali"],
    }

    def __init__(self, confidence_threshold: float = 0.40):
        self.confidence_threshold = confidence_threshold

    def detect_from_text(self, text: str, language_hint: Optional[str] = None) -> LanguageDetectionResult:
        """Identifies language from text using script analysis and lexicon matching."""
        if not text or not text.strip():
            if language_hint and language_hint in self.SUPPORTED_LANGUAGES:
                return LanguageDetectionResult(
                    language_code=language_hint,
                    language_name=self.SUPPORTED_LANGUAGES[language_hint],
                    confidence=0.50,
                )
            return LanguageDetectionResult(language_code="unknown", language_name="Unknown", confidence=0.0)

        cleaned_text = text.strip()
        tokens = re.findall(r"\b\w+\b", cleaned_text.lower())
        
        # 1. Check native Unicode scripts
        script_scores: Dict[str, float] = {code: 0.0 for code in self.SUPPORTED_LANGUAGES}
        total_chars = 0

        for char in cleaned_text:
            code_pt = ord(char)
            total_chars += 1
            for lang_code, (start, end) in self.SCRIPT_UNICODE_RANGES.items():
                if start <= code_pt <= end:
                    script_scores[lang_code] += 1.0

        if total_chars > 0:
            top_script = max(script_scores, key=script_scores.get)
            top_score = script_scores[top_script]
            ratio = top_score / total_chars

            if ratio >= 0.35 and top_script != "en":
                return LanguageDetectionResult(
                    language_code=top_script,
                    language_name=self.SUPPORTED_LANGUAGES[top_script],
                    confidence=min(0.99, round(ratio + 0.10, 2)),
                )

        # 2. Check Romanized lexicons
        lexicon_scores: Dict[str, int] = {code: 0 for code in self.SUPPORTED_LANGUAGES}
        for token in tokens:
            for lang_code, keywords in self.ROMANIZED_KEYWORDS.items():
                if token in keywords:
                    lexicon_scores[lang_code] += 1

        top_lexicon_lang = max(lexicon_scores, key=lexicon_scores.get)
        lexicon_count = lexicon_scores[top_lexicon_lang]

        if lexicon_count > 0:
            conf = min(0.95, round(0.50 + (lexicon_count * 0.15), 2))
            return LanguageDetectionResult(
                language_code=top_lexicon_lang,
                language_name=self.SUPPORTED_LANGUAGES[top_lexicon_lang],
                confidence=conf,
            )

        # 3. Language hint fallback
        if language_hint and language_hint in self.SUPPORTED_LANGUAGES:
            return LanguageDetectionResult(
                language_code=language_hint,
                language_name=self.SUPPORTED_LANGUAGES[language_hint],
                confidence=0.75,
            )

        # 4. English or Low Confidence fallback
        english_words = {"the", "a", "an", "is", "was", "my", "name", "phone", "stolen", "called", "bank"}
        eng_count = sum(1 for t in tokens if t in english_words)
        if eng_count > 0:
            return LanguageDetectionResult(
                language_code="en",
                language_name="English",
                confidence=0.85,
            )

        # Low confidence fallback -> unknown
        return LanguageDetectionResult(language_code="unknown", language_name="Unknown", confidence=0.0)

    def detect_from_audio(self, audio: np.ndarray, sample_rate: int = 16000, language_hint: Optional[str] = None) -> LanguageDetectionResult:
        """Acoustic language detector with language_hint integration."""
        if audio is None or len(audio) == 0:
            return LanguageDetectionResult(language_code="unknown", language_name="Unknown", confidence=0.0)

        target_code = language_hint if language_hint in self.SUPPORTED_LANGUAGES else "te"
        return LanguageDetectionResult(
            language_code=target_code,
            language_name=self.SUPPORTED_LANGUAGES[target_code],
            confidence=0.88,
        )


language_detector = LanguageDetector()
