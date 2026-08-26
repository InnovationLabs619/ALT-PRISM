"""
PRISM Backend Application Entry Point
======================================
Police Investigation System Management - Local Self-Hosted Speech Pipeline API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.backend.api.v1.endpoints.auth import router as auth_router
from apps.backend.api.v1.endpoints.health import router as health_router
from apps.backend.api.v1.endpoints.sessions import router as sessions_router
from apps.backend.api.v1.endpoints.speech import router as speech_router
from apps.backend.api.v1.endpoints.stream import router as stream_router
from apps.backend.core.config import settings
from ml.inference.engine import InferenceEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load & warm local AI models once
    print("[PRISM API] Initializing self-hosted inference engine...")
    engine = InferenceEngine()
    engine.initialize_models()
    print("[PRISM API] All self-hosted AI models loaded and ready.")
    yield
    # Shutdown: Clean up any resources
    print("[PRISM API] Shutting down PRISM services...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="PRISM - Police Investigation System Management (100% Self-Hosted Indic ASR & Translation)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Internal development & tactical network
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 Routers
api_v1_prefix = settings.API_V1_STR

app.include_router(health_router, prefix=api_v1_prefix)
app.include_router(auth_router, prefix=api_v1_prefix)
app.include_router(speech_router, prefix=api_v1_prefix)
app.include_router(stream_router, prefix=api_v1_prefix)
app.include_router(sessions_router, prefix=api_v1_prefix)


@app.get("/")
async def root():
    return {
        "service": "PRISM",
        "description": "Police Investigation System Management API",
        "status": "operational",
        "docs": "/docs",
        "health_check": f"{settings.API_V1_STR}/health",
        "ai_health": f"{settings.API_V1_STR}/health/ai",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.backend.main:app", host="0.0.0.0", port=8000, reload=False)
