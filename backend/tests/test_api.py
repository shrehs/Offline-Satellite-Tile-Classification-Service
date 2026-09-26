"""
Integration tests against the running FastAPI app (via TestClient), backed
by a synthetic model + throwaway SQLite DB (see conftest.py). Matches the
5-test plan from DESIGN.md, minus the offline test (that's in
test_classifier.py since it's about model loading, not the API).
"""

from tests.conftest import CLASSES, count_prediction_rows


def test_valid_tile_returns_200_and_persists(test_client, sample_png_bytes):
    """Test 1: PNG -> 200 -> prediction returned -> DB row exists."""
    rows_before = count_prediction_rows(test_client._db_path)

    response = test_client.post(
        "/classify",
        files={"file": ("tile.png", sample_png_bytes, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "id" in body
    assert "predicted_class" in body

    rows_after = count_prediction_rows(test_client._db_path)
    assert rows_after == rows_before + 1


def test_invalid_image_returns_4xx_and_does_not_persist(test_client):
    """Test 2: bad file -> 4xx -> no prediction stored."""
    rows_before = count_prediction_rows(test_client._db_path)

    garbage_bytes = b"this is not an image, just some bytes"
    response = test_client.post(
        "/classify",
        files={"file": ("not_an_image.png", garbage_bytes, "image/png")},
    )

    assert 400 <= response.status_code < 500

    rows_after = count_prediction_rows(test_client._db_path)
    assert rows_after == rows_before  # nothing was written


def test_output_validity(test_client, sample_png_bytes):
    """Test 3: predicted_class in the 7 classes, confidence in [0,1], status is valid."""
    response = test_client.post(
        "/classify",
        files={"file": ("tile.png", sample_png_bytes, "image/png")},
    )
    body = response.json()

    assert body["predicted_class"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["status"] in ("accepted", "uncertain")

    # probabilities should be a full distribution over all 7 classes
    assert set(body["probabilities"].keys()) == set(CLASSES)
    prob_sum = sum(body["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01  # softmax rounding tolerance


def test_persistence_roundtrip(test_client, sample_png_bytes):
    """Test 4: API response ID -> DB (via GET /predictions/{id}) -> same prediction."""
    post_response = test_client.post(
        "/classify",
        files={"file": ("tile.png", sample_png_bytes, "image/png")},
    )
    posted = post_response.json()
    prediction_id = posted["id"]

    get_response = test_client.get(f"/predictions/{prediction_id}")
    assert get_response.status_code == 200
    fetched = get_response.json()

    assert fetched["predicted_class"] == posted["predicted_class"]
    assert fetched["confidence"] == posted["confidence"]
    assert fetched["status"] == posted["status"]
    assert fetched["input_sha256"] == posted["input_sha256"]


def test_get_nonexistent_prediction_returns_404(test_client):
    response = test_client.get("/predictions/999999")
    assert response.status_code == 404