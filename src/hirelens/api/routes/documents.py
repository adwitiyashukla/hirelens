from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from hirelens.api.db.repository import DocumentRepository
from hirelens.api.deps import SessionDep
from hirelens.api.schemas import (
    DocumentOut,
    DocumentText,
    RejectedUpload,
    UploadResponse,
    UploadResult,
)
from hirelens.ingest import IngestionError, read_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md"}


@router.post("", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    session: AsyncSession = SessionDep,
) -> UploadResponse:
    accepted: list[UploadResult] = []
    rejected: list[RejectedUpload] = []
    repository = DocumentRepository(session)

    for upload in files:
        filename = upload.filename or "unnamed"
        suffix = Path(filename).suffix.lower()

        if suffix not in SUPPORTED_SUFFIXES:
            rejected.append(
                RejectedUpload(
                    filename=filename,
                    reason=f"Unsupported type '{suffix}'. Accepted: "
                    + ", ".join(sorted(SUPPORTED_SUFFIXES)),
                )
            )
            continue

        payload = await upload.read()
        if not payload:
            rejected.append(RejectedUpload(filename=filename, reason="File is empty"))
            continue
        if len(payload) > MAX_UPLOAD_BYTES:
            rejected.append(
                RejectedUpload(
                    filename=filename,
                    reason=f"File is {len(payload) / 1e6:.1f}MB, over the "
                    f"{MAX_UPLOAD_BYTES / 1e6:.0f}MB limit",
                )
            )
            continue

        try:
            source = _ingest(payload, filename, suffix)
        except IngestionError as exc:
            rejected.append(RejectedUpload(filename=filename, reason=str(exc)))
            continue
        except Exception as exc:
            logger.exception("failed to ingest %s", filename)
            rejected.append(RejectedUpload(filename=filename, reason=f"Could not read: {exc}"))
            continue

        if source.is_probably_scanned:
            rejected.append(
                RejectedUpload(
                    filename=filename,
                    reason="Almost no extractable text. This looks like a scanned "
                    "document with no text layer; OCR is not supported yet.",
                )
            )
            continue

        document, created = await repository.upsert(
            source,
            raw_bytes=payload,
            content_type=upload.content_type or "application/octet-stream",
        )
        accepted.append(
            UploadResult(document=DocumentOut.model_validate(document), created=created)
        )

    return UploadResponse(uploaded=accepted, rejected=rejected)


def _ingest(payload: bytes, filename: str, suffix: str):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(payload)
        source = read_document(path)
    return source.model_copy(update={"filename": filename})


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    limit: int = 50, offset: int = 0, session: AsyncSession = SessionDep
) -> list[DocumentOut]:
    documents = await DocumentRepository(session).list(limit=min(limit, 200), offset=offset)
    return [DocumentOut.model_validate(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str, session: AsyncSession = SessionDep) -> DocumentOut:
    document = await DocumentRepository(session).get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return DocumentOut.model_validate(document)


@router.get("/{document_id}/text", response_model=DocumentText)
async def get_document_text(document_id: str, session: AsyncSession = SessionDep) -> DocumentText:
    document = await DocumentRepository(session).get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    return DocumentText(
        document_id=document.id,
        filename=document.filename,
        page_count=document.page_count,
        text=document.text,
        blocks=(document.blocks_json or {}).get("blocks", []),
    )


@router.get("/{document_id}/raw")
async def get_document_raw(document_id: str, session: AsyncSession = SessionDep) -> Response:
    document = await DocumentRepository(session).get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    if document.raw_bytes is None:
        raise HTTPException(status_code=404, detail="Original file was not retained")

    return Response(
        content=document.raw_bytes,
        media_type=document.content_type,
        headers={"Content-Disposition": f'inline; filename="{document.filename}"'},
    )
