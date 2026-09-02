import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import uvicorn

from .config import (CORS_ORIGINS, FRONTEND_DIR, SAMPLES_DIR, MODEL_PATH,
                     MODEL_ARCH, IMG_SIZE, THRESHOLD_PATH)
from .database import init_db
from .routers import diagnostics


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("Database initialized.")
    yield


app = FastAPI(
    title="DermaScan AI — Diagnostic Engine",
    description="EfficientNet-B3 skin cancer classification API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(diagnostics.router)

# Serve frontend and the reference lesion images (paths are absolute so the
# server can be started from any working directory).
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health_check():
    """Reports the actual serving configuration.

    Deliberately does not touch the predictor: the model is loaded lazily on the
    first analysis, and a health check should not pull 123 MB into memory. It
    reports whether that load has happened rather than triggering it.
    """
    from .routers import diagnostics

    return {
        "status": "ok",
        "version": app.version,
        "model": {
            "architecture": MODEL_ARCH,
            "checkpoint": os.path.basename(MODEL_PATH),
            "checkpoint_present": os.path.exists(MODEL_PATH),
            "thresholds_present": os.path.exists(THRESHOLD_PATH),
            "input_size": IMG_SIZE,
            "loaded": diagnostics._predictor is not None,
        },
    }


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8088, reload=True)
