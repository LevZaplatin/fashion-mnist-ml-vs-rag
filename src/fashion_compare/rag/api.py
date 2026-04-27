from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from fashion_compare.config import get_settings
from fashion_compare.labels import labels_metadata
from fashion_compare.models.registry import all_mode_names
from fashion_compare.rag.retrieve import predict_rag
from fashion_compare.utils.image_io import base64_to_image, load_image_bytes, pixel_array_to_numpy


class PredictRequest(BaseModel):
    image_base64: str | None = None
    pixels: list[list[float]] | None = None


app = FastAPI(title="Fashion-MNIST RAG Retrieval Classifier API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "system": "rag"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {"system": "rag", "labels": labels_metadata(), "modes": all_mode_names()}


async def _parse_request_image(request: Request, image: UploadFile | None) -> Any:
    if image is not None:
        return load_image_bytes(await image.read())
    body = await request.json()
    payload = PredictRequest.model_validate(body)
    if payload.image_base64:
        return base64_to_image(payload.image_base64)
    if payload.pixels is not None:
        return pixel_array_to_numpy(payload.pixels)
    raise HTTPException(status_code=422, detail="Provide multipart image, image_base64, or pixels")


@app.post("/predict")
async def predict(
    request: Request,
    image: UploadFile | None = File(default=None),
    mode: str | None = Query(default=None),
    top_k: int | None = Query(default=None, ge=1, le=100),
) -> dict[str, Any]:
    settings = get_settings()
    mode = mode or settings.embedding_mode
    top_k = top_k or settings.top_k
    try:
        parsed = await _parse_request_image(request, image)
        return predict_rag(parsed, mode=mode, top_k=top_k)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
