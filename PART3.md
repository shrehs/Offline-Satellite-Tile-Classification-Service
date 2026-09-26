# Part 3 — Reasoning

## 1. If the classifier is wrong ~30% of the time, what would you do? How would you decide whether it is still useful?

I would not decide based on the overall error rate alone. I would first measure **where and how the errors occur**, then compare that behavior with the intended use of the predictions.

### Step 1: Measure the errors

I would evaluate the model on representative unseen data and inspect:

- Overall accuracy and macro F1
- Per-class precision and recall
- Confusion matrix
- Confidence distribution
- Which classes are being confused with each other
- Whether errors are concentrated in particular types of imagery

For this assignment, the selected ResNet18 achieved **88.1% validation accuracy and 88.1% validation macro F1**. On the frozen 210-tile evaluation set, it achieved **85.7% accuracy and 85.7% macro F1**.

I would use the same type of analysis if performance degraded to around 70%.

### Step 2: Understand the operational consequence

A 30% error rate can have very different consequences depending on what the predictions are used for.

If predictions are being used to **prioritize or assist an analyst**, the classifier may still be useful if it reduces manual effort and uncertain cases can be reviewed.

If predictions are being used to **automatically trigger a high-impact decision**, the same error rate may not be acceptable.

Therefore, I would define an acceptable error level based on the cost of a wrong prediction, not accuracy alone.

### Step 3: Use uncertainty to avoid treating every prediction equally

The system separates classification from abstention.

The model always selects its top class using `argmax`, but a low-confidence prediction can be marked `uncertain` and routed for review.

For ResNet18, a validation-only confidence sweep showed:

| Threshold | Accepted | Accuracy among accepted |
|---:|---:|---:|
| 0.3 | 99.0% | 88.94% |
| 0.4 | 95.7% | 91.04% |
| 0.5 | 90.0% | 93.12% |
| **0.6** | **81.4%** | **95.91%** |
| 0.7 | 73.3% | 96.10% |
| 0.8 | 60.0% | 96.83% |

The current operating point is 0.6, selected from validation data only. At that point, the model accepts approximately 81% of validation tiles with approximately 96% accuracy among accepted predictions.

This does not mean that the softmax score is perfectly calibrated. It is an operating threshold that trades coverage against reliability.

### Decision

I would therefore treat the classifier as useful if it provides measurable value for the intended workflow, including after accounting for the cost of reviewing uncertain predictions. If its errors are systematic or concentrated in an important class, I would investigate the data/model before simply accepting the 70% accuracy.

---

## 2. The system runs offline with no live monitoring. How would you know a month later that it is still working correctly?

I would separate **system health** from **prediction correctness**.

A service can still be running and returning HTTP 200 responses while producing incorrect predictions because of a changed model artifact, preprocessing behavior, corrupted inputs, or a shift in the incoming data.

### Operational health

The service should persist or expose locally available information such as:

- Number of tiles received
- Successful classifications
- Validation failures
- Inference failures
- Storage failures
- Inference latency
- Accepted vs uncertain predictions
- Confidence distribution
- Predicted-class distribution
- Model version and checksum

Because the machine is offline, these signals should not depend on a remote monitoring service.

### Reproducibility

Each stored prediction records the model version and model checksum. This makes it possible to determine exactly which model artifact produced an older prediction.

The preprocessing configuration should also be tied to the model version because changing preprocessing can change predictions even when the weights are unchanged.

### Deterministic golden tests

I would retain a small set of known test tiles with expected outputs.

A periodic local health check can run these tiles through the complete pipeline:

```text
input tile
    ↓
validation
    ↓
preprocessing
    ↓
model
    ↓
postprocessing
    ↓
expected result
```

If the expected behavior changes unexpectedly, that indicates a software/model/environment regression even if the API itself is still responding.

### Distribution checks

I would also inspect prediction statistics over time. For example, a sudden change in:

- class distribution,
- confidence distribution,
- uncertain rate, or
- inference latency

could indicate a change in the incoming data or system behavior.

This is especially important because a model can remain technically operational while its input distribution changes.

---

## 3. Tiles are arriving and stored results are wrong. What do you debug first, and in what order?

I would debug from the **input boundary toward persistence**, looking for the first point where the actual value differs from the expected value.

The investigation would follow the same pipeline used by the service:

```text
Raw PNG
   ↓
Input validation
   ↓
Image decode
   ↓
Preprocessing
   ↓
Model input tensor
   ↓
ResNet18
   ↓
Logits
   ↓
Softmax / argmax
   ↓
Prediction object
   ↓
SQLite write
   ↓
SQLite read / API response
```

### 1. Reproduce the problem

Take the original tile and reproduce the prediction outside the database/query path.

If the same tile produces the same wrong prediction directly through the classifier, the problem is upstream of persistence.

### 2. Verify the exact input

Check:

- Input SHA-256
- Tile identity
- Image dimensions
- RGB channels
- Whether the stored/source file is actually the tile that was expected

This catches cases where the wrong image was supplied or associated with the wrong tile ID.

### 3. Check preprocessing

Compare the inference preprocessing with the preprocessing used during training:

- Resize/shape
- Channel ordering
- Data type
- Scaling
- Normalization

A preprocessing mismatch can produce systematically wrong predictions even when the model weights are correct.

### 4. Check the model artifact

Verify:

- Model version
- Model checksum
- Correct class ordering
- Correct checkpoint
- Model loaded successfully

This is why the prediction record stores model metadata.

### 5. Compare raw model output

Run the same input directly through the model and inspect the logits/probability vector.

This tells me whether the error is actually coming from the model or from postprocessing.

### 6. Check postprocessing

Verify:

- Softmax is applied correctly
- `argmax` maps to the correct class
- Class-index-to-name mapping is correct
- Confidence is the maximum score
- The uncertainty threshold is applied correctly

### 7. Check persistence

If the direct classifier output is correct but the API/database result is wrong, inspect:

- Database write
- Serialization of probabilities
- Stored class/confidence fields
- Database read/query mapping

At this point the model is no longer the primary suspect.

### 8. Check concurrency/batching if applicable

If the problem only appears under load, I would investigate shared state, batching, request handling, or race conditions.

The main principle is:

> **Find the first divergence in the pipeline rather than debugging the entire system at once.**

---

## 4. What is the weakest part of your design? What would break first?

The weakest part is the **model's ability to generalize to future imagery**, rather than the basic HTTP or database layer.

The current system is evaluated on a particular dataset distribution. A production system could receive imagery that differs from the development data. For example, the geographic distribution, image characteristics, class balance, or other properties of incoming tiles could change.

The backend could still be completely healthy:

```text
HTTP request → 200
model loaded → yes
SQLite write → success
```

while prediction quality has degraded.

### Why this matters

The frozen evaluation result gives evidence about performance on the held-out evaluation set, but it cannot guarantee performance on every future distribution.

That is why the design includes:

- Model versioning
- Model checksums
- Confidence/uncertainty tracking
- Class-distribution monitoring
- Golden-tile checks
- Representative validation/error analysis

If performance degrades, I would first establish whether the incoming data has shifted and identify the failure pattern. Depending on the evidence, the response could involve new training data, fine-tuning, recalibration, a different model, or a changed abstention policy.

I would **not** solve a model-quality problem by adding backend infrastructure alone.

---

## Summary

The main design principle across all four questions is to make uncertainty and failure observable rather than hiding them.

The system therefore:

1. Measures model errors instead of relying only on aggregate accuracy.
2. Separates classification from an explicit abstention policy.
3. Persists model/version information for offline reproducibility.
4. Uses local health checks because the environment cannot rely on live monitoring.
5. Debugs predictions by tracing the first divergence through the pipeline.
6. Treats model/domain mismatch and distribution shift as the primary long-term reliability risk.
