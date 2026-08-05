import os
import re
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from config.settings import settings
from src.api.schemas import IngestRequest, IngestResponse, UploadResponse
from src.api.dependencies import get_rag_service
from src.service import RAGPipelineService

router = APIRouter(tags=["Ingestion"])

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".doc", ".docx", ".html", ".htm", ".csv", ".xlsx", ".xls"}


def _ensure_doc_dirs() -> None:
    Path(settings.DOCS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.UPLOADED_DOCS_DIR).mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    base = os.path.basename(name).strip()
    base = re.sub(r"[^\w\s.\-]", "_", base)
    return base or "upload.txt"


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_document(req: IngestRequest, service: RAGPipelineService = Depends(get_rag_service)):
    try:
        res = service.ingest_source(
            source=req.source,
            document_id=req.document_id,
            chunking_strategy=req.chunking_strategy or settings.CHUNKING_STRATEGY,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            allowed_roles=req.allowed_roles
        )
        return IngestResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    folder: str = Query("Uploaded-Docs", description="Uploaded-Docs or Docs"),
    auto_ingest: bool = Query(True, description="Index the document after upload"),
    service: RAGPipelineService = Depends(get_rag_service),
):
    _ensure_doc_dirs()

    if folder not in (settings.UPLOADED_DOCS_DIR, settings.DOCS_DIR, "Uploaded-Docs", "Docs"):
        raise HTTPException(status_code=400, detail="folder must be 'Uploaded-Docs' or 'Docs'")

    target_dir = settings.UPLOADED_DOCS_DIR if folder in ("Uploaded-Docs", settings.UPLOADED_DOCS_DIR) else settings.DOCS_DIR
    filename = _safe_filename(file.filename or "upload.txt")
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    dest = Path(target_dir) / filename
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        counter = 1
        while dest.exists():
            dest = Path(target_dir) / f"{stem}_{counter}{suffix}"
            counter += 1

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    dest.write_bytes(content)
    saved_path = str(dest).replace("\\", "/")

    ingestion_result = None
    if auto_ingest:
        chunking = "recursive" if ext in {".pdf", ".doc", ".csv", ".xlsx", ".xls"} else settings.CHUNKING_STRATEGY
        try:
            ingestion_result = service.ingest_source(
                source=saved_path,
                chunking_strategy=chunking,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=f"File saved to {saved_path} but cannot be indexed: {e}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"File saved to {saved_path} but ingestion failed: {e}",
            )

    return UploadResponse(
        filename=dest.name,
        saved_path=saved_path,
        folder=target_dir,
        ingested=auto_ingest,
        ingestion=IngestResponse(**ingestion_result) if ingestion_result else None,
    )


@router.get("/documents")
def list_documents(service: RAGPipelineService = Depends(get_rag_service)):
    _ensure_doc_dirs()
    stats = service.vector_store.get_stats()
    uploaded = [f.name for f in Path(settings.UPLOADED_DOCS_DIR).iterdir() if f.is_file()]
    docs = [f.name for f in Path(settings.DOCS_DIR).iterdir() if f.is_file()]
    return {
        "stats": stats,
        "total_chunks_indexed": len(service._all_chunk_documents),
        "uploaded_docs": uploaded,
        "docs_folder": docs,
    }
