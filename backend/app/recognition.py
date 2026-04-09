import logging
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class FaceRecognitionEngine:
    """
    ArcFace recognition via InsightFace buffalo_l model.

    The Android app sends pre-aligned 112x112 crops.
    We still run InsightFace detection as a safety net when the API
    is called directly (e.g. curl, Postman, web frontend).
    """

    def __init__(self):
        logger.info("Loading InsightFace buffalo_l...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        # Use a det_size divisible by 32 to prevent stride mismatch in InsightFace config 
        self.app.prepare(ctx_id=0, det_size=(128, 128))
        logger.info("Model ready")

    def get_embedding(self, image_bytes: bytes) -> np.ndarray | None:
        """
        Accepts raw image bytes (JPEG/PNG).
        Returns L2-normalised 512D embedding, or None if no face found.

        The Android client sends a pre-aligned 112x112 JPEG crop.
        For direct API usage (testing, web frontend), full-size images also work —
        InsightFace runs its own detection internally.
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Could not decode image bytes")
                return None

            faces = self.app.get(img)
            if not faces:
                logger.warning(f"No face detected (image shape: {img.shape})")
                return None

            # If multiple faces, pick largest (shouldn't happen for aligned crops)
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.embedding

            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                logger.warning("Zero-norm embedding — skipping")
                return None

            return (emb / norm).astype(np.float32)

        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised vectors.
        Both inputs must already be L2-normalised (norm ≈ 1.0).
        Result range: -1.0 (opposite) to 1.0 (identical).
        Typical same-person: 0.45–0.85. Different person: 0.0–0.35.
        """
        return float(np.dot(a, b))

    def best_match_score(self, query: np.ndarray, stored: list[np.ndarray]) -> float:
        """Return the best cosine similarity across a list of stored embeddings."""
        if not stored:
            return 0.0
        return max(self.cosine_similarity(query, e) for e in stored)

    def batch_similarity(self, query: np.ndarray, stored: list[np.ndarray]) -> float:
        """Backward-compatible alias for older callers/tests."""
        return self.best_match_score(query, stored)
