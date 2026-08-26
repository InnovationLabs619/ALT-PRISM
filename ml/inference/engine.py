"""
PRISM Central Inference Engine
==============================
Manages model lifecycle (STARTING, LOADING, READY, BUSY, ERROR, UNAVAILABLE),
device allocation, warmup, end-to-end speech processing pipeline, and diagnostics.
"""

import enum
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ml.inference.asr_provider import ASRProvider, IndicConformerASRProvider
from ml.inference.code_switch import CodeSwitchAnalyzer, CodeSwitchResult
from ml.inference.translation_provider import IndicTrans2Provider, TranslationProvider
from ml.preprocessing.audio import AudioMetadata, AudioPreprocessor


class EngineState(str, enum.Enum):
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class TimingMetrics:
    audio_duration: float
    preprocessing_latency: float
    asr_latency: float
    code_switch_latency: float
    translation_latency: float
    total_latency: float


@dataclass
class ProcessingResponse:
    session_id: str
    language: str
    language_confidence: float
    is_code_switched: bool
    languages: List[str]
    transcript: str
    english_translation: str
    segments: List[Dict[str, Any]]
    timing: TimingMetrics
    models: Dict[str, str]
    audio_quality: Dict[str, Any]
    status: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timing"] = asdict(self.timing)
        return d


class InferenceEngine:
    """Singleton Inference Engine powering PRISM."""

    _instance: Optional["InferenceEngine"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        asr_provider: Optional[ASRProvider] = None,
        translation_provider: Optional[TranslationProvider] = None,
        code_switch_analyzer: Optional[CodeSwitchAnalyzer] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
        device: str = "auto",
    ):
        if getattr(self, "_initialized", False):
            return

        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.state = EngineState.STARTING
        self.init_start_time = time.time()

        self.preprocessor = preprocessor or AudioPreprocessor()
        self.code_switch_analyzer = code_switch_analyzer or CodeSwitchAnalyzer()

        # Instantiate providers with auto-fallback
        self.asr_provider = asr_provider or IndicConformerASRProvider(device=self.device)
        self.translation_provider = translation_provider or IndicTrans2Provider(device=self.device)

        self.total_inferences_served = 0
        self.total_audio_seconds_processed = 0.0
        self.last_error = None
        self._initialized = True

    def initialize_models(self) -> None:
        """Loads and warms up all models once."""
        self.state = EngineState.LOADING
        print(f"[PRISM Engine] Initializing models on {self.device.upper()}...")
        try:
            # Load ASR
            self.asr_provider.load_model()
            # Load Translation
            self.translation_provider.load_model()

            # Warmup
            print("[PRISM Engine] Running model warmup routines...")
            self.asr_provider.warm_up()
            self.translation_provider.warm_up()

            self.state = EngineState.READY
            print("[PRISM Engine] AI Engine status: READY")
        except Exception as e:
            self.state = EngineState.ERROR
            self.last_error = str(e)
            print(f"[PRISM Engine Error] Failed to initialize models: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Returns structured health and model version metadata."""
        vram_mb = None
        if self.device == "cuda" and torch.cuda.is_available():
            vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)

        return {
            "status": self.state.value,
            "device": self.device,
            "vram_allocated_mb": vram_mb,
            "asr": {
                "status": "ready" if self.asr_provider and self.asr_provider.is_loaded else "not_loaded",
                "model": self.asr_provider.model_name if self.asr_provider else "none",
                "version": self.asr_provider.model_version if self.asr_provider else "none",
            },
            "translation": {
                "status": "ready" if self.translation_provider and self.translation_provider.is_loaded else "not_loaded",
                "model": self.translation_provider.model_name if self.translation_provider else "none",
                "version": self.translation_provider.model_version if self.translation_provider else "none",
            },
            "metrics": {
                "inferences_count": self.total_inferences_served,
                "audio_seconds_processed": round(self.total_audio_seconds_processed, 2),
                "uptime_seconds": round(time.time() - self.init_start_time, 2),
            },
        }

    def process_speech(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        session_id: str,
        language_hint: Optional[str] = None,
        target_language: str = "en",
    ) -> ProcessingResponse:
        """
        Full unified speech pipeline:
        Audio -> Preprocessing -> ASR -> Code-Switch Analysis -> Translation -> Output.
        """
        overall_start = time.time()
        self.state = EngineState.BUSY

        # 1. Preprocessing
        t0 = time.time()
        try:
            clean_audio, audio_meta = self.preprocessor.process(audio_input)
            t_preproc = round(time.time() - t0, 4)
        except Exception as e:
            self.state = EngineState.READY
            return ProcessingResponse(
                session_id=session_id,
                language=language_hint or "unknown",
                language_confidence=0.0,
                is_code_switched=False,
                languages=[],
                transcript="",
                english_translation="",
                segments=[],
                timing=TimingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, round(time.time() - overall_start, 4)),
                models={"asr": self.asr_provider.model_name, "translation": self.translation_provider.model_name},
                audio_quality={"snr_db": 0.0, "quality_score": 0.0, "channels": 1, "sample_rate": 16000},
                status="AUDIO_VALIDATION_FAILED",
                error_message=f"Audio preprocessing failed: {str(e)}",
            )

        if not audio_meta.is_valid_speech:
            self.state = EngineState.READY
            return ProcessingResponse(
                session_id=session_id,
                language=language_hint or "unknown",
                language_confidence=0.0,
                is_code_switched=False,
                languages=[],
                transcript="",
                english_translation="",
                segments=[],
                timing=TimingMetrics(audio_meta.duration_sec, t_preproc, 0.0, 0.0, 0.0, round(time.time() - overall_start, 4)),
                models={"asr": self.asr_provider.model_name, "translation": self.translation_provider.model_name},
                audio_quality={
                    "snr_db": audio_meta.snr_db,
                    "quality_score": audio_meta.quality_score,
                    "channels": audio_meta.channels,
                    "sample_rate": audio_meta.sample_rate,
                },
                status="NO_SPEECH_DETECTED",
                error_message="Audio duration or volume below minimum speech threshold.",
            )

        # 2. ASR Inference
        t0 = time.time()
        asr_res = self.asr_provider.transcribe(
            clean_audio,
            sample_rate=audio_meta.sample_rate,
            language_hint=language_hint,
        )
        t_asr = round(time.time() - t0, 4)

        if not asr_res.text:
            self.state = EngineState.READY
            return ProcessingResponse(
                session_id=session_id,
                language=asr_res.language,
                language_confidence=0.0,
                is_code_switched=False,
                languages=[asr_res.language],
                transcript="",
                english_translation="",
                segments=[],
                timing=TimingMetrics(audio_meta.duration_sec, t_preproc, t_asr, 0.0, 0.0, round(time.time() - overall_start, 4)),
                models={"asr": self.asr_provider.model_name, "translation": self.translation_provider.model_name},
                audio_quality={
                    "snr_db": audio_meta.snr_db,
                    "quality_score": audio_meta.quality_score,
                    "channels": audio_meta.channels,
                    "sample_rate": audio_meta.sample_rate,
                },
                status="ASR_NO_TRANSCRIPT",
            )

        # 3. Code-Switch Analysis
        t0 = time.time()
        cs_res: CodeSwitchResult = self.code_switch_analyzer.analyze(
            asr_res.text, language_hint=asr_res.language
        )
        t_cs = round(time.time() - t0, 4)

        # 4. Translation
        t0 = time.time()
        indic_langs = [l for l in cs_res.languages if l != "en"]
        source_lang = indic_langs[0] if indic_langs else cs_res.primary_language
        if source_lang == "en" and asr_res.language and asr_res.language != "en":
            source_lang = asr_res.language

        trans_res = self.translation_provider.translate(
            text=asr_res.text,
            source_lang=source_lang,
            target_lang=target_language,
        )
        t_trans = round(time.time() - t0, 4)

        total_lat = round(time.time() - overall_start, 4)
        self.state = EngineState.READY

        # Update metrics
        self.total_inferences_served += 1
        self.total_audio_seconds_processed += audio_meta.duration_sec

        return ProcessingResponse(
            session_id=session_id,
            language=cs_res.primary_language,
            language_confidence=cs_res.confidence,
            is_code_switched=cs_res.is_code_switched,
            languages=cs_res.languages,
            transcript=asr_res.text,
            english_translation=trans_res.translated_text,
            segments=[s.to_dict() if hasattr(s, "to_dict") else asdict(s) for s in asr_res.segments],
            timing=TimingMetrics(
                audio_duration=audio_meta.duration_sec,
                preprocessing_latency=t_preproc,
                asr_latency=t_asr,
                code_switch_latency=t_cs,
                translation_latency=t_trans,
                total_latency=total_lat,
            ),
            models={
                "asr": self.asr_provider.model_name,
                "asr_version": self.asr_provider.model_version,
                "translation": self.translation_provider.model_name,
                "translation_version": self.translation_provider.model_version,
                "pipeline_version": "prism-mvp-1.0",
            },
            audio_quality={
                "snr_db": audio_meta.snr_db,
                "quality_score": audio_meta.quality_score,
                "channels": audio_meta.channels,
                "sample_rate": audio_meta.sample_rate,
            },
            status="success",
        )
