"""
app/repository.py

SQLite persistence for predictions. Deliberately plain - a thin wrapper
around sqlite3, no ORM. One table, matching the schema from DESIGN.md:

    predictions
    ────────────────────────
    id
    tile_id
    input_sha256
    predicted_class
    confidence
    status
    probabilities_json
    model_version
    model_checksum
    inference_ms
    created_at

Usage:
    repo = PredictionRepository("predictions.db")
    repo.init_db()
    pred_id = repo.save(tile_id="tile.png", result=classifier_result)
    row = repo.get(pred_id)
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    probabilities_json TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_checksum TEXT NOT NULL,
    inference_ms REAL NOT NULL,
    created_at TEXT NOT NULL
);
"""


class PredictionRepository:
    def __init__(self, db_path: Union[str, Path] = "predictions.db"):
        self.db_path = str(db_path)

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, tile_id: str, result: dict) -> int:
        """
        `result` is the dict returned by Classifier.predict() / predict_bytes():
        predicted_class, confidence, status, probabilities, model_version,
        model_checksum, inference_ms, input_sha256.
        """
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    tile_id, input_sha256, predicted_class, confidence, status,
                    probabilities_json, model_version, model_checksum,
                    inference_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tile_id,
                    result["input_sha256"],
                    result["predicted_class"],
                    result["confidence"],
                    result["status"],
                    json.dumps(result["probabilities"]),
                    result["model_version"],
                    result["model_checksum"],
                    result["inference_ms"],
                    created_at,
                ),
            )
            assert cursor.lastrowid is not None  # guaranteed after a successful INSERT
            return cursor.lastrowid

    def get(self, prediction_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_by_sha256(self, input_sha256: str) -> Optional[dict]:
        """Useful for dedup checks / idempotency, and for Test 4 (persistence roundtrip)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE input_sha256 = ? ORDER BY id DESC LIMIT 1",
                (input_sha256,),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def list_all(self, limit: int = 100) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["probabilities"] = json.loads(d.pop("probabilities_json"))
        return d


if __name__ == "__main__":
    # Manual smoke test - run directly to sanity check the repo in isolation,
    # same pattern as classifier.py's __main__ block.
    repo = PredictionRepository("smoke_test.db")
    repo.init_db()

    fake_result = {
        "predicted_class": "Forest",
        "confidence": 0.87,
        "status": "accepted",
        "probabilities": {
            "AnnualCrop": 0.01, "Forest": 0.87, "Highway": 0.02,
            "Industrial": 0.03, "Residential": 0.02, "River": 0.01, "SeaLake": 0.04,
        },
        "model_version": "resnet18-eurosat-v1",
        "model_checksum": "deadbeef",
        "inference_ms": 12.3,
        "input_sha256": "abc123",
    }

    pred_id = repo.save(tile_id="smoke_test_tile.png", result=fake_result)
    print("Saved with id:", pred_id)
    print("Fetched:", repo.get(pred_id))