from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from fashion_compare.classifier.predict import classifier_metadata, load_classifier, predict_image
from fashion_compare.config import get_settings
from fashion_compare.utils.image_io import base64_to_image, load_image_bytes, pixel_array_to_numpy


class PredictRequest(BaseModel):
    image_base64: str | None = None
    pixels: list[list[float]] | None = None


app = FastAPI(title="Fashion-MNIST CNN Classifier API")
_model: Any = None
_metadata: dict[str, Any] = {}


@app.on_event("startup")
def startup() -> None:
    global _model, _metadata
    settings = get_settings()
    if settings.model_path.exists():
        _model, _metadata = load_classifier()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "system": "classifier", "model_loaded": _model is not None}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return classifier_metadata(_metadata)


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
async def predict(request: Request, image: UploadFile | None = File(default=None)) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Classifier checkpoint is not available")
    try:
        parsed = await _parse_request_image(request, image)
        return predict_image(parsed, model=_model)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
