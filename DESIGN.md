# GalaxEye Backend Engineer, ML Systems — Design Note

## 1. Overview

This design describes an offline tile-classification service for satellite imagery. The system receives a 64×64 RGB PNG tile, runs a locally stored image classifier, persists the prediction and inference metadata, and exposes the result through an API.

The design intentionally keeps the system small and explainable. The assignment asks for a working slice rather than a production-scale distributed platform, so the implementation uses a single service and SQLite while making model versioning, reproducibility, uncertainty, and debugging explicit.

The assignment specifies an isolated/offline environment and satellite imagery. The submitted dataset contains 64×64 RGB PNG tiles from a subset of EuroSAT, with seven classes: AnnualCrop, Forest, Highway, Industrial, Residential, River, and SeaLake.

## 2. Assumptions

### Input

- The service receives a 64×64 RGB PNG image.
- Input pixels are 8-bit (`uint8`).
- The working dataset contains RGB imagery only; although EuroSAT originates from Sentinel-2 imagery, this implementation does **not** assume that raw multispectral bands are available.
- The input must be decodable as a valid PNG and have the expected dimensions/channels.

### Runtime

- The machine is isolated/offline.
- The model artifact and Python dependencies are available locally.
- Inference is CPU-based.
- No hosted model API or external network call is required.

### Storage and workload

- SQLite is sufficient for the take-home's single-node, moderate-workload implementation.
- Prediction records should remain queryable after inference.
- The system should preserve enough metadata to reproduce and debug a prediction.

### Product behavior

- A classifier always produces a top class using `argmax`.
- Separately, the service uses model confidence as an uncertainty signal.
- Low-confidence predictions are marked `uncertain` rather than being silently treated as equally reliable.
- The initial confidence threshold is a validation-derived operating point, not a universally calibrated probability.

## 3. End-to-End Architecture

```text
                    Offline Machine
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  Client / Analyst                                             │
│       │                                                       │
│       │ POST /classify (PNG tile)                             │
│       ▼                                                       │
│  ┌───────────────┐                                            │
│  │ Input         │  Validate PNG, size, RGB channels          │
│  │ Validation    │                                            │
│  └───────┬───────┘                                            │
│          ▼                                                    │
│  ┌───────────────┐                                            │
│  │ Preprocessing │  Decode → tensor → training normalization │
│  └───────┬───────┘                                            │
│          ▼                                                    │
│  ┌───────────────┐                                            │
│  │ ResNet18      │  Local CPU inference                       │
│  │ Classifier    │                                            │
│  └───────┬───────┘                                            │
│          ▼                                                    │
│  ┌───────────────┐                                            │
│  │ Postprocess   │  Softmax → argmax + confidence             │
│  │ / Abstention  │  confidence < threshold → uncertain       │
│  └───────┬───────┘                                            │
│          │                                                    │
│          ├───────────────┐                                    │
│          ▼               ▼                                    │
│  ┌───────────────┐  ┌───────────────┐                         │
│  │ SQLite        │  │ JSON Response │                         │
│  │ Prediction    │  │               │                         │
│  │ Record        │  │ class/status  │                         │
│  └───────────────┘  └───────────────┘                         │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

The same prediction record is returned to the caller and persisted. This avoids a situation where an API response says one thing while the stored record contains another.

## 4. Tile Processing Flow

For every incoming tile:

1. **Validate input**
   - Confirm that the uploaded content is a valid PNG.
   - Confirm expected 64×64 dimensions.
   - Confirm RGB channels.
   - Reject malformed or unsupported input before inference.

2. **Identify the input**
   - Compute a SHA-256 hash of the input bytes.
   - Use the hash as a reproducibility/debugging identifier.
   - The hash also helps identify duplicate inputs.

3. **Preprocess**
   - Decode the image.
   - Convert to the tensor representation expected by the model.
   - Apply the same normalization used during model training.
   - Do not introduce a different preprocessing pipeline at inference time.

4. **Run local inference**
   - Load the frozen ResNet18 model.
   - Run CPU inference with gradients disabled.
   - Produce seven class logits.

5. **Postprocess**
   - Apply softmax to obtain the class score vector.
   - Select the predicted class using `argmax`.
   - Record the maximum score as the model's confidence/uncertainty signal.
   - If confidence is below the selected threshold, mark the result `uncertain`.

6. **Persist**
   - Store the prediction, confidence, status, model metadata, input hash, latency, and timestamp.
   - Store the complete seven-class score vector where practical, rather than only the winning class.

7. **Return**
   - Return the prediction ID, class, confidence, status, model version, and inference metadata.

## 5. Model Selection

Three candidate models were evaluated on the training/validation split:

| Model | Validation Accuracy | Macro F1 | CPU Latency / Image |
|---|---:|---:|---:|
| SimpleCNN | 72.86% | 72.09% | 1.31 ms |
| MobileNetV3 | 81.43% | 80.97% | 1.78 ms |
| **ResNet18** | **88.10%** | **88.13%** | **2.17 ms** |

ResNet18 was selected because it provided the strongest validation accuracy and macro F1. Its measured CPU latency was only about 0.9 ms/image higher than MobileNetV3 in the local experiment, so the additional inference cost was small relative to the validation-performance difference.

The final evaluation set was kept frozen and was evaluated only once.

### Frozen evaluation

- Accuracy: **85.71%**
- Macro F1: **85.71%**
- 210 evaluation tiles
- 30 tiles per class

The validation-to-frozen-evaluation difference is expected to be treated as a generalization measurement rather than used to tune the model.

## 6. Uncertainty and Abstention

Classification and abstention are deliberately separate.

### Classification

The classifier always selects:

```text
predicted_class = argmax(class_scores)
```

There is no rule saying that a score below 0.5 means "no class."

### Abstention

The maximum class score is used as an uncertainty signal:

```text
if confidence >= threshold:
    status = "accepted"
else:
    status = "uncertain"
```

The threshold was selected using validation data only.

### Validation threshold sweep

For ResNet18:

| Threshold | Accepted Tiles | Accuracy Among Accepted |
|---:|---:|---:|
| 0.3 | 99.0% | 88.94% |
| 0.4 | 95.7% | 91.04% |
| **0.5** | **90.0%** | **93.12%** |
| **0.6** | **81.4%** | **95.91%** |
| 0.7 | 73.3% | 96.10% |
| 0.8 | 60.0% | 96.83% |

The initial operating point is **0.6**, giving approximately 81% coverage and 96% accuracy among accepted validation predictions.

This threshold should be understood as an operating policy rather than a claim that the softmax score is perfectly calibrated. In a real deployment, the threshold would depend on the cost of a wrong automated classification versus sending a tile for human review.

The frozen evaluation set was not used for threshold selection.

## 7. What Should Be Stored?

A prediction record should contain enough information to answer: *What did the system predict, why did it produce that result, and which model produced it?*

Suggested record:

| Field | Purpose |
|---|---|
| `id` | Unique prediction identifier |
| `tile_id` | Caller/application tile identifier if available |
| `input_sha256` | Identifies exact input bytes |
| `predicted_class` | Top predicted class |
| `confidence` | Maximum model score |
| `status` | `accepted` or `uncertain` |
| `probabilities_json` | Complete seven-class score vector |
| `model_version` | Identifies model artifact |
| `model_checksum` | Detects model-artifact changes |
| `inference_ms` | Local inference latency |
| `created_at` | Timestamp |

Storing only the predicted class would make later debugging difficult. For example, if a River tile is incorrectly classified as Highway, the full score vector can show whether Highway was barely ahead of River or whether the model was strongly confident in the wrong class.

The original image itself can be retained if operational requirements require it. For this assignment, retaining the image is not necessary for the prediction database, provided the input hash and source tile identity are preserved.

## 8. Querying Semantics

The core working slice is classification plus persistence. Analyst querying is intentionally lightweight.

Useful query patterns include:

```text
GET /predictions/{id}
GET /predictions?label=Forest
GET /predictions?status=uncertain
GET /predictions?min_confidence=0.8
```

These queries operate on stored prediction metadata; they do not rerun inference.

This separation matters because analysts should be able to inspect historical results without changing them by invoking the model again.

For a larger deployment, query requirements would determine whether SQLite remains appropriate or whether a server database/search layer is needed.

## 9. Failure Handling

Failures should be visible rather than converted into misleading predictions.

### Invalid input

Examples:

- malformed PNG
- wrong dimensions
- unsupported channel count

Return a validation error and do not create a prediction record.

### Model failure

Examples:

- model artifact missing
- incompatible model artifact
- runtime inference exception

Return an error and record sufficient application logs/metrics to diagnose the failure.

### Storage failure

If inference succeeds but persistence fails, the API should not report the operation as successfully stored. The service should distinguish:

```text
inference_success != persistence_success
```

This distinction is important for operational correctness.

## 10. Observability and Offline Operation

Because the target environment is offline, the system cannot depend on a remote monitoring service.

The service should therefore make important operational information available locally.

Useful metrics/logs include:

- number of tiles received
- successful classifications
- validation failures
- inference failures
- storage failures
- inference latency
- confidence distribution
- accepted vs uncertain counts
- predicted-class distribution
- model version/checksum

However, operational health alone does not prove prediction correctness.

A small set of deterministic "golden" test tiles can be retained locally. A periodic offline verification can run those tiles through the same preprocessing and model pipeline and compare the results with expected outputs.

This can detect cases where the service is still running but preprocessing, model files, dependencies, or postprocessing behavior have changed.

## 11. Model Versioning and Reproducibility

Every persisted prediction should identify the model version.

A model version should correspond to a specific local artifact, with a checksum recorded alongside it.

For example:

```text
model_version = resnet18-eurosat-v1
model_checksum = <sha256>
```

This allows two predictions made months apart to be traced to the exact model artifact that produced them.

When a new model is introduced, it should receive a new model version rather than silently replacing the old artifact.

The preprocessing configuration should be versioned with the model because changing normalization or input transformations can change predictions even when the model weights are unchanged.

## 12. Storage Choice and Scaling Trade-off

### SQLite for the assignment

Advantages:

- zero external service
- works fully offline
- easy to inspect
- simple deployment
- transactional persistence
- sufficient for the thin slice

Limitations:

- limited concurrent-write scalability
- not intended as a distributed storage system
- operational querying becomes less suitable at larger scale

### If workload grows

If the real system needs multiple service instances, high write concurrency, or large-scale analyst queries, the persistence layer could be replaced with PostgreSQL or another appropriate database.

The application should keep storage access behind a small repository/data-access layer so that this change does not require rewriting the classifier.

The assignment does not justify introducing distributed infrastructure such as Kafka, Kubernetes, Redis, or multiple microservices into the thin slice.

## 13. Key Design Trade-offs

### Accuracy vs latency

ResNet18 is slower than SimpleCNN and MobileNetV3, but the measured latency remains low on CPU and the validation performance is materially stronger.

### Coverage vs reliability

A lower confidence threshold accepts more predictions but also accepts more errors. A higher threshold improves accuracy among accepted predictions while increasing the uncertain/review workload.

The chosen 0.6 operating point is therefore a product/system decision, not simply an ML parameter.

### Full score vector vs storage size

Storing all seven class scores costs little and significantly improves debugging and analysis compared with storing only the winning class.

### SQLite vs production database

SQLite minimizes complexity and is appropriate for a single-node offline slice. A production deployment with higher concurrency would justify a stronger database.

### Original image vs metadata-only storage

Keeping original imagery makes debugging easier but increases storage and potentially introduces data-retention concerns. For the thin slice, the input hash plus prediction metadata is sufficient.

## 14. What Could Break First?

The weakest part is likely to be **model/domain mismatch rather than the HTTP or database layer**.

A healthy backend can still produce poor results if incoming imagery differs from the data used to train and validate the classifier. This can happen because of changes in geographic regions, image characteristics, class distributions, acquisition conditions, or other distribution shifts.

The system should therefore make model versioning, confidence monitoring, class-distribution monitoring, and representative validation easy to perform.

If the model's error rate becomes unacceptable, the response should be driven by evidence from error analysis and representative new data rather than by assuming that a backend change will solve the problem.

## 15. Open Questions for GalaxEye

The following questions would materially affect the production design:

1. **Can the system abstain?**
   - Is an uncertain result acceptable, or must every tile receive a class?

2. **What is the analyst workflow?**
   - What queries or filters are expected after predictions are stored?

3. **How are predictions used?**
   - Are they informational for analysts, or do they trigger automated downstream decisions?

4. **What workload should be expected?**
   - Approximate tiles per batch/day and expected concurrency would determine whether SQLite remains appropriate.

5. **Should original imagery be retained?**
   - If so, for how long and with what storage constraints?

6. **How are model updates handled?**
   - Is there an expected approval/versioning process for new model artifacts?

7. **What are the operational latency requirements?**
   - Is processing expected tile-by-tile, in batches, or near-real-time?

These answers would determine which parts of the thin slice need to evolve for production.

## 16. Scope of the Working Slice

The implementation intentionally focuses on one complete vertical path:

```text
PNG tile
   ↓
validation
   ↓
preprocessing
   ↓
local ResNet18 inference
   ↓
class + confidence + uncertainty status
   ↓
SQLite persistence
   ↓
JSON response
```

Batch ingestion, advanced analyst querying, distributed storage, model serving infrastructure, and production monitoring are left outside the working slice. They are represented in the design where they affect interfaces and trade-offs, but are not implemented unnecessarily for the assignment.
