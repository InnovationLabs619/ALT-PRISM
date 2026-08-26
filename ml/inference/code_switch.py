"""
PRISM Code-Switching Analyzer
=============================
Explicitly analyzes code-mixing and script transitions between Indian languages
and English (e.g. "Na phone railway station daggara theft ayyindi").
Provides token-level, segment-level, and utterance-level code-switch metrics.
"""

import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CodeSwitchSpan:
    text: str
    language: str
    confidence: float
    start_char: int
    end_char: int
    script: str


@dataclass
class CodeSwitchResult:
    is_code_switched: bool
    primary_language: str
    languages: List[str]
    confidence: float
    segments: List[Dict]
    switch_points_count: int
    code_switch_ratio: float

    def to_dict(self) -> Dict:
        return asdict(self)


class CodeSwitchAnalyzer:
    """Analyzes Indian multilingual and code-switched text."""

    SCRIPT_RANGES = {
        "Telugu": (0x0C00, 0x0C7F, "te"),
        "Devanagari": (0x0900, 0x097F, "hi"),  # Shared by Hindi, Marathi
        "Tamil": (0x0B80, 0x0BFF, "ta"),
        "Kannada": (0x0C80, 0x0CFF, "kn"),
        "Malayalam": (0x0D00, 0x0D7F, "ml"),
        "Bengali": (0x0980, 0x09FF, "bn"),
        "Gujarati": (0x0A80, 0x0AFF, "gu"),
        "Odia": (0x0B00, 0x0B7F, "or"),
        "Gurmukhi": (0x0A00, 0x0A7F, "pa"),
        "Latin": (0x0041, 0x007A, "en"),
    }

    # High-frequency Romanized Indic functional words for transliterated detection
    ROMANIZED_INDIC_LEXICON: Dict[str, Set[str]] = {
        "te": {
            "na", "nenu", "maku", "ma", "daggara", "ninna", "eeroju", "cheppanu", "poyindi",
            "ayyindi", "unnadu", "vachadu", "velladu", "police", "station", "garu", "valla",
            "dabbulu", "bangaaram", "kallu", "chusa", "chesadu", "undi", "vundhi", "leka"
        },
        "hi": {
            "mera", "meri", "mere", "hum", "kal", "aaj", "gaya", "gayi", "chori", "hua", "hui",
            "tha", "thi", "the", "police", "chowki", "paas", "paisa", "gaadi", "dekha", "hai"
        },
        "ta": {
            "en", "enakku", "nethu", "innikku", "pochu", "thiruttu", "aachu", "irukku", "vandharu",
            "ponaru", "kaasu", "thirudan", "paathen", "sonnen"
        },
        "kn": {
            "nanna", "nanage", "innu", "ivattu", "aayithu", "kaddaru", "idhe", "biddare", "nodide",
            "police", "hana", "hege"
        },
        "mr": {
            "maza", "maze", "aamhi", "kal", "aaj", "gela", "zala", "hota", "hoti", "paise", "baghitla"
        },
        "bn": {
            "amar", "amra", "kal", "aj", "geche", "hoyeche", "churi", "chilo", "dekhechi", "taka"
        },
        "ml": {
            "ente", "enikku", "innale", "innu", "poyi", "aayi", "undayirunnu", "kandu", "panam"
        }
    }

    COMMON_ENGLISH_WORDS: Set[str] = {
        "the", "a", "an", "phone", "mobile", "bike", "car", "station", "railway", "theft", "stolen",
        "missing", "wallet", "bag", "gold", "chain", "money", "cash", "road", "street", "house",
        "accident", "threat", "fraud", "card", "bank", "account", "officer", "sir", "madam",
        "yesterday", "today", "night", "morning", "evening", "near", "hospital", "bus", "stop"
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

    def detect_script(self, char: str) -> Optional[Tuple[str, str]]:
        """Returns (script_name, language_code) for a single character."""
        code_point = ord(char)
        for script_name, (start, end, lang) in self.SCRIPT_RANGES.items():
            if start <= code_point <= end:
                return script_name, lang
        return None

    def analyze_token(self, token: str, default_indic_hint: Optional[str] = None) -> Tuple[str, str, float]:
        """
        Determines the script, language, and confidence of an individual word token.
        Returns: (script_name, language_code, confidence).
        """
        cleaned = re.sub(r"[^\w\s]", "", token).strip().lower()
        if not cleaned:
            return "Punctuation", "unknown", 0.0

        # Check native Unicode script first
        script_counts: Dict[str, int] = {}
        for c in token:
            res = self.detect_script(c)
            if res:
                s_name, _ = res
                script_counts[s_name] = script_counts.get(s_name, 0) + 1

        if script_counts:
            dominant_script = max(script_counts, key=script_counts.get)
            if dominant_script != "Latin":
                lang = next(
                    l for s, (_, _, l) in self.SCRIPT_RANGES.items() if s == dominant_script
                )
                conf = script_counts[dominant_script] / max(1, len(token))
                return dominant_script, lang, min(1.0, conf + 0.2)

        # Romanized / Latin script token analysis
        if cleaned in self.COMMON_ENGLISH_WORDS:
            return "Latin", "en", 0.95

        # Check Romanized Indic lexicons
        matched_indic_langs = []
        for lang_code, vocab in self.ROMANIZED_INDIC_LEXICON.items():
            if cleaned in vocab:
                matched_indic_langs.append(lang_code)

        if matched_indic_langs:
            if default_indic_hint and default_indic_hint in matched_indic_langs:
                return "Latin-Indic", default_indic_hint, 0.90
            return "Latin-Indic", matched_indic_langs[0], 0.85

        # Latin morphological suffix heuristics (Telugu/Hindi transliteration patterns)
        if cleaned.endswith(("ayyindi", "poyindi", "unnadu", "garu", "daggara", "valla", "lo", "ki")):
            return "Latin-Indic", "te", 0.85
        if cleaned.endswith(("gaya", "gayi", "karega", "walon", "wala", "chahiye")):
            return "Latin-Indic", "hi", 0.85
        if cleaned.endswith(("aachu", "irukku", "pudichu")):
            return "Latin-Indic", "ta", 0.85

        # Default fallback for Latin word
        if default_indic_hint and default_indic_hint != "en":
            # If uncertain Latin token in Indic context
            return "Latin", "en", 0.70

        return "Latin", "en", 0.80

    def analyze(self, text: str, language_hint: Optional[str] = None) -> CodeSwitchResult:
        """
        Analyzes a full transcript string for code-switching patterns, language boundaries,
        and code-switch proportion.
        """
        if not text or not text.strip():
            return CodeSwitchResult(
                is_code_switched=False,
                primary_language=language_hint or "en",
                languages=[language_hint or "en"],
                confidence=1.0,
                segments=[],
                switch_points_count=0,
                code_switch_ratio=0.0,
            )

        tokens = re.finditer(r"\S+", text)
        token_spans: List[CodeSwitchSpan] = []
        detected_languages: Dict[str, int] = {}
        language_weights: Dict[str, float] = {}

        for match in tokens:
            word = match.group()
            start = match.start()
            end = match.end()

            script, lang, conf = self.analyze_token(word, language_hint)
            token_spans.append(
                CodeSwitchSpan(
                    text=word,
                    language=lang,
                    confidence=round(conf, 2),
                    start_char=start,
                    end_char=end,
                    script=script,
                )
            )

            if lang != "unknown":
                detected_languages[lang] = detected_languages.get(lang, 0) + 1
                language_weights[lang] = language_weights.get(lang, 0.0) + conf

        # Calculate switch points
        switch_points = 0
        for i in range(1, len(token_spans)):
            prev_l = token_spans[i - 1].language
            curr_l = token_spans[i].language
            if prev_l != "unknown" and curr_l != "unknown" and prev_l != curr_l:
                switch_points += 1

        total_tokens = len(token_spans)
        unique_active_langs = [l for l, count in detected_languages.items() if count >= 1 and l != "unknown"]

        # Primary language determined by aggregate weight
        if language_weights:
            primary_lang = max(language_weights, key=language_weights.get)
        else:
            primary_lang = language_hint or "te"

        is_code_switched = len(unique_active_langs) > 1 and switch_points >= 1
        non_primary_count = sum(c for l, c in detected_languages.items() if l != primary_lang and l != "unknown")
        code_switch_ratio = round(non_primary_count / max(1, total_tokens), 3)

        # Merge adjacent spans of identical language for segments
        merged_segments = []
        if token_spans:
            cur_seg = {
                "text": token_spans[0].text,
                "language": token_spans[0].language,
                "script": token_spans[0].script,
                "confidence": token_spans[0].confidence,
                "start_char": token_spans[0].start_char,
                "end_char": token_spans[0].end_char,
            }

            for span in token_spans[1:]:
                if span.language == cur_seg["language"]:
                    cur_seg["text"] += " " + span.text
                    cur_seg["end_char"] = span.end_char
                else:
                    merged_segments.append(cur_seg)
                    cur_seg = {
                        "text": span.text,
                        "language": span.language,
                        "script": span.script,
                        "confidence": span.confidence,
                        "start_char": span.start_char,
                        "end_char": span.end_char,
                    }
            merged_segments.append(cur_seg)

        overall_confidence = 0.90 if total_tokens > 2 else 0.70
        if not unique_active_langs:
            overall_confidence = 0.40

        return CodeSwitchResult(
            is_code_switched=is_code_switched,
            primary_language=primary_lang,
            languages=unique_active_langs if unique_active_langs else [primary_lang],
            confidence=overall_confidence,
            segments=merged_segments,
            switch_points_count=switch_points,
            code_switch_ratio=code_switch_ratio,
        )
