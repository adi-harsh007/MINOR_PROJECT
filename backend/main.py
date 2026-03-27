from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from .database import init_db
from .routers import diagnostics

app = FastAPI(
    title="DermaScan AI — Diagnostic Engine",
    description="EfficientNet-B3 skin cancer classification API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diagnostics.router)

# Serve frontend
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")

@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("frontend/index.html") # Or just return 204 No Content

@app.on_event("startup")
def on_startup():
    init_db()
    print("Database initialized.")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "engine": "EfficientNet-B3", "version": "3.0.0"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8088, reload=True)
