"""
PRISM Single-Command System Launcher
====================================
Launches the FastAPI backend on port 8000 and the React tactical dashboard on port 5173.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    print("=" * 70)
    print("   PRISM // POLICE INVESTIGATION SYSTEM MANAGEMENT")
    print("   100% Self-Hosted Indic ASR & Translation Platform")
    print("=" * 70)

    # 1. Start FastAPI Backend
    print("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "apps.backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(BASE_DIR),
    )

    # 2. Start Frontend Dev Server
    print("[2/2] Starting React Tactical Dashboard on http://localhost:5173...")
    frontend_dir = BASE_DIR / "apps" / "frontend"
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(frontend_dir),
    )

    print("\n" + "=" * 70)
    print("PRISM System Active & Ready!")
    print(" - Dashboard: http://localhost:5173")
    print(" - API Docs:  http://localhost:8000/docs")
    print(" - AI Health: http://localhost:8000/api/v1/health/ai")
    print("Press Ctrl+C to terminate all services.")
    print("=" * 70 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping PRISM services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("PRISM services successfully stopped.")


if __name__ == "__main__":
    main()
