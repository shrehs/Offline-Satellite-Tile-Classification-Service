"""
app/main.py

FastAPI wrapper around the classifier + repository. This file should
stay thin - validation, wiring, error handling. No model logic here
(that's classifier.py), no SQL here (that's repository.py).

Run with:
    uvicorn app.main:app --reload

Flow for POST /classify:
    validate image -> classifier.predict_bytes() -> repository.save() -> JSON response
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import io

from app.classifier import Classifier
from app.repository import PredictionRepository

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
MODEL_CHECKPOINT = BASE_DIR / "model" / "resnet18_eurosat_v1.pth"
MODEL_METADATA = BASE_DIR / "model" / "model_metadata.json"
DB_PATH = BASE_DIR / "predictions.db"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB - generous for a single tile, blocks abuse

# Loaded once at startup, not per-request - this is the expensive part
# (model weights + repo connection setup), so it shouldn't happen per-call.
classifier: Classifier = None
repository: PredictionRepository = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, repository
    classifier = Classifier(checkpoint_path=MODEL_CHECKPOINT, metadata_path=MODEL_METADATA)
    repository = PredictionRepository(db_path=DB_PATH)
    repository.init_db()
    yield
    # no explicit teardown needed - sqlite3 connections are opened/closed
    # per-call in repository.py, nothing persistent to release here


app = FastAPI(title="Tile Classifier API", lifespan=lifespan)


@app.get("/health")
def health():
    """Basic liveness check - also useful for confirming the model loaded
    without making a real prediction request."""
    return {
        "status": "ok",
        "model_version": classifier.model_version if classifier else None,
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # --- validate content type up front ---
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # --- validate it's actually a decodable image, not just a matching content-type header ---
    try:
        Image.open(io.BytesIO(contents)).verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="File is not a valid image")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not process image")

    # --- run inference ---
    try:
        result = classifier.predict_bytes(contents)
    except Exception as e:
        # Inference failure is a 500, not a 4xx - the input was valid, something
        # internal broke (e.g. unexpected tensor shape, device error).
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # --- persist ---
    prediction_id = repository.save(tile_id=file.filename, result=result)

    return {
        "id": prediction_id,
        **result,
    }


@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int):
    """Not in the original required-endpoint list, but trivial to add and
    directly supports Test 4 (API response ID -> DB -> same prediction)."""
    row = repository.get(prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return row