"""
PRISM Configuration & Environment Settings
==========================================
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Set cache directories on workspace drive (134+ GB available)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = BASE_DIR / ".cache"
HF_HOME = CACHE_DIR / "huggingface"
TORCH_HOME = CACHE_DIR / "torch"

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["TORCH_HOME"] = str(TORCH_HOME)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(TORCH_HOME, exist_ok=True)


class Settings(BaseSettings):
    PROJECT_NAME: str = "PRISM"
    PROJECT_FULL_NAME: str = "Police Investigation System Management"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "prism-secure-jwt-secret-key-internal-police-dept-2026"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ALGORITHM: str = "HS256"

    # Hardware & Model Settings
    AI_DEVICE: str = "auto"  # 'auto', 'cpu', 'cuda'
    ASR_MODEL_NAME: str = "openai/whisper-tiny"  # Fast, accurate local self-hosted weights
    TRANSLATION_MODEL_NAME: str = "facebook/nllb-200-distilled-600M"

    # Audio & Preprocessing
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    MAX_AUDIO_DURATION_SEC: float = 600.0

    # Storage Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    SAMPLES_DIR: Path = BASE_DIR / "data" / "samples"
    MODELS_DIR: Path = BASE_DIR / "models"
    REPORTS_DIR: Path = BASE_DIR / "ml" / "reports"

    # CORS
    BACKEND_CORS_ORIGINS: list = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "allow"


settings = Settings()
