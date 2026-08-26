"""
PRISM Health & Observability Router
===================================
Exposes system health, AI model status, and runtime metrics.
"""

from fastapi import APIRouter
from ml.inference.engine import InferenceEngine

router = APIRouter(tags=["Health & Diagnostics"])


@router.get("/health")
async def get_system_health():
    return {
        "status": "healthy",
        "service": "PRISM Police Investigation System Management",
        "version": "1.0.0",
        "environment": "production-internal",
    }


@router.get("/health/ai")
async def get_ai_engine_health():
    engine = InferenceEngine()
    return engine.get_health_status()


@router.get("/metrics")
async def get_metrics():
    engine = InferenceEngine()
    health = engine.get_health_status()
    return {
        "metrics": health["metrics"],
        "device": health["device"],
        "asr_model": health["asr"]["model"],
        "translation_model": health["translation"]["model"],
    }
