"""
DermaScan AI — Unified Startup Script
Launches the FastAPI backend which serves both the API and frontend.
Usage: python start.py
"""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("\n  Starting DermaScan AI on http://localhost:8088 ...")
    print("  API Docs : http://localhost:8088/docs")
    print("  Press Ctrl+C to stop.\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "backend.main:app",
             "--host", "0.0.0.0", "--port", "8088", "--reload"],
            cwd=ROOT,
        )
    except KeyboardInterrupt:
        print("\n  Server stopped.\n")

if __name__ == "__main__":
    main()
