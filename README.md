# Offline Satellite Tile Classification Service

A small backend service that classifies satellite image tiles into one of
seven EuroSAT-style land-cover classes, using a trained ResNet18 checkpoint.
The service serves predictions and persists them locally — it does not
train or retrain any model at runtime.

---

## What this is

- A `POST /classify` endpoint that accepts a single image tile and returns
  a predicted land-cover class, a confidence score, and an accept/uncertain
  status.
- Every prediction is persisted to a local SQLite database, along with the
  model version and checksum that produced it.
- Designed to run **offline**, with no dependency on a live monitoring
  service or network access at inference time.

Model selection, the multi-model comparison (SimpleCNN vs. ResNet18 vs.
MobileNetV3), and the validation confidence/abstention sweep are documented
separately in [`DESIGN.md`](./DESIGN.md) and [`PART3.md`](./PART3.md) — this
README focuses on running and operating the service itself.

---

## Architecture

```text
POST /classify
      ↓
validate image (content-type, decodability, size)
      ↓
SHA-256 of input
      ↓
preprocessing (resize, normalize)
      ↓
ResNet18 (frozen backbone + trained classifier head)
      ↓
softmax → argmax
      ↓
confidence ≥ threshold?  →  accepted / uncertain
      ↓
SQLite
      ↓
JSON response
```

```text
backend/
├── app/
│   ├── classifier.py   # standalone inference component - model load + predict
│   ├── repository.py   # SQLite persistence layer
│   └── main.py          # FastAPI app - wires classifier + repository together
├── model/
│   ├── resnet18_eurosat_v1.pth   # trained checkpoint (not included - see below)
│   └── model_metadata.json        # classes, input size, normalization, threshold
├── tests/
│   ├── conftest.py
│   ├── test_classifier.py
│   └── test_api.py
├── requirements.txt
└── pytest.ini
```

`classifier.py` has no FastAPI or SQLite dependency and can be run and
tested in isolation. `main.py` stays thin: validation, wiring, error
handling — no model logic and no SQL of its own.

---

## Model

The service uses a **ResNet18** transfer-learning model:

- Backbone frozen at ImageNet-pretrained weights; only the final
  classification head was trained.
- 7 classes: `AnnualCrop`, `Forest`, `Highway`, `Industrial`,
  `Residential`, `River`, `SeaLake`.
- Input: 64×64 RGB, normalized with ImageNet mean/std.
- Model architecture, class ordering, input size, and normalization are
  all recorded in `model/model_metadata.json` and loaded from there at
  startup — none of it is hardcoded separately in the serving code.

Training, the baseline comparison against SimpleCNN and MobileNetV3, and
the validation confidence sweep that produced the abstention threshold
below were done separately from this service and are **not** re-run by
the API. See `DESIGN.md` for that process and `PART3.md` for the
reasoning behind the operating threshold.

The trained checkpoint (`resnet18_eurosat_v1.pth`) is treated as a frozen
artifact: the service loads it read-only at startup and never modifies it.

---

## Setup

```powershell
cd backend
pip install -r requirements.txt
```

Place the trained checkpoint at `backend/model/resnet18_eurosat_v1.pth`.
`model/model_metadata.json` is already included and describes the classes,
input size, normalization, and confidence threshold the checkpoint expects.

---

## Run

```powershell
cd backend
uvicorn app.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Health check: `GET /health`

The model is loaded once at startup (not per-request), so a healthy
`/health` response confirms the checkpoint loaded correctly before you
send any real tiles.

---

## API example

```powershell
curl.exe -X POST "http://127.0.0.1:8000/classify" -F "file=@path\to\tile.png"
```

### Example response

```json
{
  "id": 1,
  "predicted_class": "AnnualCrop",
  "confidence": 0.6698,
  "status": "accepted",
  "probabilities": {
    "AnnualCrop": 0.6698,
    "Forest": 0.0032,
    "Highway": 0.0439,
    "Industrial": 0.0225,
    "Residential": 0.0119,
    "River": 0.064,
    "SeaLake": 0.1849
  },
  "model_version": "resnet18-eurosat-v1",
  "model_checksum": "6a4803f422b445a5709d11ee745a9ab987156cac3023aec2661456441ba9d34f",
  "inference_ms": 15.244,
  "input_sha256": "20b9de6b92a06c2779825910e1d134028689ebb912fe59dd97bca8f33bf9345f"
}
```

`status` is `"accepted"` when confidence is at or above the configured
threshold (`0.6`), and `"uncertain"` otherwise. The prediction is always
the argmax class — the threshold only decides whether that prediction is
flagged as trustworthy, not what the prediction itself is.

Other endpoints:

- `GET /predictions/{id}` — fetch a previously stored prediction by ID
- `GET /health` — liveness check, also reports the loaded model version

---

## Tests

```powershell
cd backend
pytest -v
```

8 tests covering:

- Valid tile → 200 → prediction returned → DB row exists
- Invalid image → 4xx → nothing persisted
- Output validity (predicted class is one of the 7 classes, confidence in
  `[0, 1]`, status is `accepted`/`uncertain`, probabilities sum to ~1)
- Persistence roundtrip (POST response ID matches what `GET /predictions/{id}`
  returns)
- 404 on a nonexistent prediction ID
- Model loading requires no network access
- `predict_bytes()` output shape
- Deterministic input/model checksums for identical inputs

Tests run against a synthetic ResNet18 (correct architecture, randomly
initialized weights) rather than the real trained checkpoint, so the suite
is self-contained and doesn't depend on shipping model weights. This means
the tests verify **plumbing and contracts** — status codes, persistence,
output shape, no-network-at-load — not that the specific trained checkpoint
classifies correctly. Model accuracy is verified separately (see below),
not by this test suite.

---

## Evaluation

The trained ResNet18 checkpoint was evaluated once, after the model and
threshold were frozen, against a held-out 210-tile evaluation set reserved
specifically for final evaluation (never used during model selection or
threshold tuning):

| Metric | Value |
|---|---|
| Accuracy | 85.7% |
| Macro F1 | 85.7% |

This number reflects argmax classification and is independent of the
abstention threshold — the threshold only affects whether a prediction is
marked `accepted` vs. `uncertain`, not what the model predicts. Full
per-class precision/recall, the confusion matrix, the model comparison
that led to choosing ResNet18, and the validation-only confidence/
abstention sweep behind the `0.6` threshold are documented in `DESIGN.md`
and `PART3.md`.

---

## Offline assumptions

- The model checkpoint is loaded from local disk at startup with
  `weights=None` passed to the base architecture — no ImageNet weights are
  downloaded at load time, only the locally saved trained state dict.
- SQLite is a local file (`backend/predictions.db`); no external database
  connection is required.
- No telemetry, external logging, or monitoring service is contacted.
- Each stored prediction records `model_version` and `model_checksum`, so a
  prediction made a month ago can be traced back to exactly which model
  artifact produced it, even without live monitoring.

---

## Known limitations

- RGB-only, 64×64 input — no raw multispectral bands.
- Classification performance depends on how similar incoming imagery is to
  the training distribution; performance on meaningfully different imagery
  is not guaranteed by this evaluation.
- SQLite is appropriate for this single-node assignment slice, not for
  high-concurrency production use.
- The 0.6 confidence threshold is an operating policy chosen from a
  validation-only sweep, not a calibrated probability — it trades coverage
  against reliability rather than representing true prediction certainty.
- The automated test suite (see Tests, above) validates plumbing and
  contracts using a synthetic model, not the accuracy of the actual trained
  checkpoint.
- Analyst/query functionality beyond `GET /predictions/{id}` is
  intentionally minimal — this is a classification service, not a review
  dashboard.
- Offline monitoring is local (logs, golden-tile checks, stored metadata)
  rather than centralized, by design, since the service is meant to run
  without network access.