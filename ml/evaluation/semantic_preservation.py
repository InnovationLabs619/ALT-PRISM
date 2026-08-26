"""
PRISM Semantic Preservation Evaluation Engine
=============================================
Evaluates whether critical incident entities (PERSON, LOCATION, DATE, TIME,
NUMBER, OBJECT, EVENT) survive speech-to-transcript-to-translation.
"""

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class EntityMatchResult:
    category: str
    ground_truth_entities: List[str]
    found_entities: List[str]
    missing_entities: List[str]
    preservation_rate: float


@dataclass
class SemanticPreservationReport:
    overall_preservation_score: float
    total_entities_expected: int
    total_entities_preserved: int
    category_scores: Dict[str, EntityMatchResult]

    def to_dict(self) -> Dict:
        return {
            "overall_preservation_score": self.overall_preservation_score,
            "total_entities_expected": self.total_entities_expected,
            "total_entities_preserved": self.total_entities_preserved,
            "category_scores": {
                k: asdict(v) for k, v in self.category_scores.items()
            },
        }


class SemanticPreservationEvaluator:
    """Computes entity preservation across PRISM translation pipelines."""

    CATEGORIES = ["PERSON", "LOCATION", "DATE", "TIME", "NUMBER", "OBJECT", "EVENT"]

    # Basic regex patterns to extract common entities when ground truth annotations are absent
    ENTITY_PATTERNS = {
        "TIME": [
            r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM|hours|hrs|o\'clock)\b",
            r"\b(?:morning|afternoon|evening|night|midnight|noon)\b",
        ],
        "DATE": [
            r"\b(?:yesterday|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
        ],
        "NUMBER": [
            r"\b\d+(?:,\d+)*(?:\.\d+)?\b",
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|lakh|crore|rupees|rs)\b",
        ],
        "OBJECT": [
            r"\b(?:phone|mobile|cellphone|iphone|android|laptop|bag|wallet|purse|money|cash|gold|chain|ring|necklace|bike|motorcycle|scooter|car|vehicle|document|documents|card|cards|otp|atm)\b"
        ],
        "LOCATION": [
            r"\b(?:station|railway\s+station|bus\s+stand|bus\s+stop|metro|junction|street|road|cross|market|shop|mall|hospital|bank|house|apartment|office|colony|nagar|vijayawada|hyderabad|bangalore|chennai|delhi|mumbai|kolkata)\b"
        ],
        "EVENT": [
            r"\b(?:theft|stolen|robbery|snatched|missing|fraud|scam|accident|crash|hit\s+and\s+run|threat|harassment|assault|attack|murder)\b"
        ],
        "PERSON": [
            r"\b(?:brother|sister|father|mother|son|daughter|wife|husband|friend|uncle|aunt|stranger|unknown\s+person|driver|officer|manager|thief|robber)\b"
        ],
    }

    def extract_candidate_entities(self, text: str) -> Dict[str, Set[str]]:
        """Heuristically extracts entities across defined categories."""
        found: Dict[str, Set[str]] = {cat: set() for cat in self.CATEGORIES}
        text_lower = text.lower()

        for category, patterns in self.ENTITY_PATTERNS.items():
            for pat in patterns:
                matches = re.finditer(pat, text_lower, re.IGNORECASE)
                for m in matches:
                    found[category].add(m.group().strip())

        return found

    def evaluate_preservation(
        self,
        translated_text: str,
        expected_entities: Dict[str, List[str]],
    ) -> SemanticPreservationReport:
        """
        Evaluates how many expected entities exist in the generated translation.
        """
        text_lower = translated_text.lower()
        category_scores: Dict[str, EntityMatchResult] = {}

        total_expected = 0
        total_preserved = 0

        for cat in self.CATEGORIES:
            expected_list = expected_entities.get(cat.lower(), []) or expected_entities.get(cat, [])
            if not expected_list:
                category_scores[cat] = EntityMatchResult(
                    category=cat,
                    ground_truth_entities=[],
                    found_entities=[],
                    missing_entities=[],
                    preservation_rate=1.0,
                )
                continue

            total_expected += len(expected_list)
            found = []
            missing = []

            for ent in expected_list:
                ent_clean = ent.strip().lower()
                # Check for exact substring match or word tokens match
                if ent_clean in text_lower or any(token in text_lower for token in ent_clean.split() if len(token) > 3):
                    found.append(ent)
                else:
                    missing.append(ent)

            total_preserved += len(found)
            rate = len(found) / max(1, len(expected_list))

            category_scores[cat] = EntityMatchResult(
                category=cat,
                ground_truth_entities=expected_list,
                found_entities=found,
                missing_entities=missing,
                preservation_rate=round(rate, 3),
            )

        overall_score = round(total_preserved / max(1, total_expected), 3) if total_expected > 0 else 1.0

        return SemanticPreservationReport(
            overall_preservation_score=overall_score,
            total_entities_expected=total_expected,
            total_entities_preserved=total_preserved,
            category_scores=category_scores,
        )
