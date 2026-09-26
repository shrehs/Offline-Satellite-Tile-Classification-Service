"""
app/classifier.py

Standalone inference component. This is the "one clean place" where
model loading and prediction happen - no FastAPI, no SQLite, no HTTP
concerns. Should be fully testable and runnable in isolation:

    classifier = Classifier("model/resnet18_eurosat_v1.pth")
    result = classifier.predict("tile.png")
"""

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


class Classifier:
    def __init__(self, checkpoint_path: Union[str, Path], metadata_path: Union[str, Path] = None):
        checkpoint_path = Path(checkpoint_path)
        metadata_path = Path(metadata_path) if metadata_path else checkpoint_path.parent / "model_metadata.json"

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.classes = self.metadata["classes"]
        self.input_size = tuple(self.metadata["input_size"])
        self.confidence_threshold = self.metadata["confidence_threshold"]
        self.model_version = self.metadata["version"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # weights=None: we load OUR trained weights below, not ImageNet ones.
        # This keeps model loading network-free (required for the offline test).
        self.model = self._build_model(len(self.classes))
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        norm = self.metadata["normalization"]
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ])

        self.model_checksum = self._sha256_file(checkpoint_path)

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _run_inference(self, img: Image.Image) -> dict:
        t0 = time.perf_counter()

        tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(probs.argmax())
        confidence = float(probs[idx])
        predicted_class = self.classes[idx]
        status = "accepted" if confidence >= self.confidence_threshold else "uncertain"

        inference_ms = (time.perf_counter() - t0) * 1000

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "status": status,
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(self.classes, probs)},
            "model_version": self.model_version,
            "model_checksum": self.model_checksum,
            "inference_ms": round(inference_ms, 3),
        }

    def predict(self, image_path: Union[str, Path]) -> dict:
        """Predict from a file path on disk."""
        img = Image.open(image_path)
        result = self._run_inference(img)
        with open(image_path, "rb") as f:
            result["input_sha256"] = self._sha256_bytes(f.read())
        return result

    def predict_bytes(self, image_bytes: bytes) -> dict:
        """Predict from raw bytes - this is what the FastAPI upload handler will call."""
        img = Image.open(io.BytesIO(image_bytes))
        result = self._run_inference(img)
        result["input_sha256"] = self._sha256_bytes(image_bytes)
        return result


if __name__ == "__main__":
    # Minimal manual smoke test - run this file directly to sanity check
    # the classifier in isolation before wiring up FastAPI.
    import sys

    if len(sys.argv) != 2:
        print("Usage: python classifier.py <path_to_tile.png>")
        sys.exit(1)

    clf = Classifier(checkpoint_path="model/resnet18_eurosat_v1.pth")
    result = clf.predict(sys.argv[1])
    print(json.dumps(result, indent=2))