from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from config.settings import settings
from src.api.routes import ingest, query, eval, health, models, memory
from src.api.dependencies import get_rag_service

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-Grade Highly Scalable Enterprise RAG Architecture API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Route Routers
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(eval.router)
app.include_router(models.router)
app.include_router(memory.router)


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.on_event("startup")
def warm_up_rag_service() -> None:
    """Initialize Chroma + BM25 once at startup to avoid request-time races."""
    get_rag_service()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
