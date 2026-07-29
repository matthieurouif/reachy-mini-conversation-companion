"""Local face recognition: OpenCV YuNet (detect) + SFace (embed).

Fork addition. Known faces live in data/faces.npz (one averaged embedding
per person), created by scripts/enroll_faces.py — or copied from the
reachy-mini-companion repo, which uses the identical format. The two small
ONNX models are auto-downloaded on first use.

Set REACHY_DATA_DIR to share a single faces database between repos.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("REACHY_DATA_DIR", _REPO_ROOT / "data"))
MODELS_DIR = DATA_DIR / "models"
FACES_DB = DATA_DIR / "faces.npz"

YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
MODEL_URLS = {
    YUNET_PATH: (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    SFACE_PATH: (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}

MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.40"))

_identifier: "FaceIdentifier | None" = None
_identifier_lock = threading.Lock()


def download_models() -> None:
    """Fetch any missing ONNX model files."""
    import requests

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in MODEL_URLS.items():
        if path.exists():
            continue
        logger.info("Downloading %s ...", path.name)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
        logger.info("Saved %s (%.1f MB)", path, len(r.content) / 1e6)


class FaceIdentifier:
    """Detects faces in frames and matches them against enrolled people."""

    def __init__(self, threshold: float = MATCH_THRESHOLD):
        """Load the ONNX models (downloading if needed) and the faces DB."""
        download_models()
        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_PATH), "", (320, 320), score_threshold=0.7
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
        self.threshold = threshold
        self.names: list[str] = []
        self.embeddings = np.zeros((0, 128), dtype=np.float32)
        self.db_mtime = 0.0
        self._infer_lock = threading.Lock()  # cv2 models are not thread-safe
        self.reload_db()

    def reload_db(self) -> None:
        """(Re)load data/faces.npz if it changed since last load."""
        if not FACES_DB.exists():
            return
        mtime = FACES_DB.stat().st_mtime
        if mtime == self.db_mtime:
            return
        db = np.load(FACES_DB)
        self.names = [str(n) for n in db["names"]]
        self.embeddings = db["embeddings"].astype(np.float32)
        self.db_mtime = mtime
        logger.info("Loaded %d enrolled people: %s", len(self.names), ", ".join(self.names))

    def detect_faces(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return YuNet detections (N x 15 array), empty array if none."""
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame_bgr)
        return faces if faces is not None else np.zeros((0, 15), dtype=np.float32)

    def embed(self, frame_bgr: np.ndarray, detection: np.ndarray) -> np.ndarray:
        """Compute an aligned 128-d SFace embedding, L2-normalized."""
        aligned = self.recognizer.alignCrop(frame_bgr, detection)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)

    def identify_detailed(self, frame_bgr: np.ndarray) -> list[dict]:
        """Return one dict per face: {name (or None), score, box (x, y, w, h)}."""
        with self._infer_lock:
            self.reload_db()
            results: list[dict] = []
            for det in self.detect_faces(frame_bgr):
                x, y, w, h = det[:4].astype(int)
                name, score = None, 0.0
                if len(self.names) > 0:
                    feat = self.embed(frame_bgr, det)
                    sims = self.embeddings @ feat
                    best = int(np.argmax(sims))
                    if sims[best] >= self.threshold:
                        name, score = self.names[best], float(sims[best])
                results.append(
                    {"name": name, "score": score, "box": (int(x), int(y), int(w), int(h))}
                )
            return results

    def identify(self, frame_bgr: np.ndarray) -> tuple[list[str], int]:
        """Return (recognized names, count of unrecognized faces) in the frame."""
        faces = self.identify_detailed(frame_bgr)
        recognized = [f["name"] for f in faces if f["name"]]
        unknown = sum(1 for f in faces if not f["name"])
        return recognized, unknown


def get_identifier() -> FaceIdentifier:
    """Return the shared FaceIdentifier, creating it on first use."""
    global _identifier
    with _identifier_lock:
        if _identifier is None:
            _identifier = FaceIdentifier()
        return _identifier
