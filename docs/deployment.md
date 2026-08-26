# PRISM On-Premise Deployment Guide
===================================
*Deployment in Air-Gapped Police Stations and Central Command Centers*

---

## 1. System Requirements

### Hardware Sizing

| Component | Minimum (Station Level) | Recommended (HQ Production) |
|---|---|---|
| **Processor** | 4-Core Intel/AMD x86_64 | 16-Core Intel Xeon / AMD EPYC |
| **System RAM** | 8 GB DDR4 | 32 GB DDR4/DDR5 |
| **GPU (Optional)** | None (runs on CPU) | 1x NVIDIA RTX 4000 / A4000 (8GB+ VRAM) |
| **Storage** | 20 GB SSD | 100 GB NVMe SSD |
| **Network** | Air-Gapped Local LAN | Department VPN / Air-Gapped Cluster |

---

## 2. Docker & Docker Compose Deployment (1-Command Launch)

PRISM is containerized to enable rapid 1-command deployment inside secure networks.

```bash
# 1. Clone the repository into station server
git clone <internal_repo_url>
cd "PROJECT PRISM"

# 2. Build and launch services in detached mode
docker-compose up -d --build

# 3. Verify health status
curl http://localhost:8000/api/v1/health/ai
```

The application will be accessible at:
- **Tactical Web Dashboard**: `http://localhost` (Port 80)
- **FastAPI Core Engine**: `http://localhost:8000` (Port 8000)
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`

---

## 3. Local Bare-Metal Setup (Windows / Linux)

### Backend Service
```bash
# 1. Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux:
source venv/bin/activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Launch Backend
python -m uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend Service
```bash
cd apps/frontend
npm install
npm run dev
```

The tactical frontend will be available at `http://localhost:5173`.
