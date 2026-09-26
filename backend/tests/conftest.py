"""
Shared fixtures.

Tests don't depend on your real trained checkpoint (resnet18_eurosat_v1.pth) -
that would make the suite slow, environment-dependent, and unable to run in CI
without shipping model weights. Instead, `synthetic_model_dir` builds a tiny
*real* ResNet18 with the correct architecture and saves a randomly-initialized
state dict via the exact same code path Classifier expects. This exercises
real load/transform/inference code, just with an untrained model - fine, since
these tests check plumbing and contracts, not accuracy.
"""

import io
import json
import sqlite3
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image


CLASSES = [
    "AnnualCrop", "Forest", "Highway", "Industrial",
    "Residential", "River", "SeaLake",
]


@pytest.fixture
def synthetic_model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Same architecture as Classifier._build_model - kept in sync deliberately.
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))

    checkpoint_path = model_dir / "test_model.pth"
    torch.save(m.state_dict(), checkpoint_path)

    metadata = {
        "model": "ResNet18",
        "version": "resnet18-eurosat-test",
        "classes": CLASSES,
        "input_size": [64, 64],
        "channels": 3,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "confidence_threshold": 0.6,
    }
    metadata_path = model_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    return checkpoint_path, metadata_path


@pytest.fixture
def sample_png_bytes():
    """A minimal valid PNG - solid color, right size range for the pipeline."""
    img = Image.new("RGB", (64, 64), color=(120, 160, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def test_client(monkeypatch, synthetic_model_dir, tmp_path):
    """
    A FastAPI TestClient wired to the synthetic checkpoint and a throwaway
    SQLite file, instead of the real model/DB paths main.py defaults to.
    """
    checkpoint_path, metadata_path = synthetic_model_dir
    db_path = tmp_path / "test_predictions.db"

    import app.main as main_module
    monkeypatch.setattr(main_module, "MODEL_CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(main_module, "MODEL_METADATA", metadata_path)
    monkeypatch.setattr(main_module, "DB_PATH", db_path)

    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as client:
        client._db_path = db_path  # stash for tests that want to inspect SQLite directly
        yield client


def count_prediction_rows(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM predictions")
        return cur.fetchone()[0]
    finally:
        conn.close()