"""
Unit tests for Classifier in isolation - no FastAPI, no SQLite.
"""

import socket

import pytest

from app.classifier import Classifier
from tests.conftest import CLASSES


def _block_all_sockets(monkeypatch):
    """Make any real network attempt fail loudly instead of silently succeeding,
    so a passing test actually proves 'no network call happened' rather than
    just 'happened not to need one this time'."""
    def _blocked(*args, **kwargs):
        raise AssertionError("Network access attempted during model loading")
    monkeypatch.setattr(socket, "socket", _blocked)


def test_classifier_loads_and_predicts_without_network(monkeypatch, synthetic_model_dir, sample_png_bytes):
    """Test 5: model loading must not make any network call."""
    checkpoint_path, metadata_path = synthetic_model_dir

    _block_all_sockets(monkeypatch)

    # Should not raise, even with sockets blocked - proves loading a saved
    # checkpoint (weights=None + load_state_dict) needs no network access.
    clf = Classifier(checkpoint_path=checkpoint_path, metadata_path=metadata_path)
    result = clf.predict_bytes(sample_png_bytes)

    assert result["predicted_class"] in CLASSES


def test_predict_bytes_output_shape(synthetic_model_dir, sample_png_bytes):
    checkpoint_path, metadata_path = synthetic_model_dir
    clf = Classifier(checkpoint_path=checkpoint_path, metadata_path=metadata_path)

    result = clf.predict_bytes(sample_png_bytes)

    expected_keys = {
        "predicted_class", "confidence", "status", "probabilities",
        "model_version", "model_checksum", "inference_ms", "input_sha256",
    }
    assert expected_keys.issubset(result.keys())
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["status"] in ("accepted", "uncertain")


def test_same_input_gives_same_checksum(synthetic_model_dir, sample_png_bytes):
    """input_sha256 should be deterministic for identical bytes - important
    for the dedup / idempotency use of get_by_sha256 in repository.py."""
    checkpoint_path, metadata_path = synthetic_model_dir
    clf = Classifier(checkpoint_path=checkpoint_path, metadata_path=metadata_path)

    result_a = clf.predict_bytes(sample_png_bytes)
    result_b = clf.predict_bytes(sample_png_bytes)

    assert result_a["input_sha256"] == result_b["input_sha256"]
    # model_checksum should also be stable across calls - same loaded weights
    assert result_a["model_checksum"] == result_b["model_checksum"]