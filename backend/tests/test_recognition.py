"""
tests/test_recognition.py
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.recognition import FaceRecognitionEngine
from app.db import Database


# ─── Recognition engine tests ────────────────────────────────────────────────

class TestCosimeSimilarity:
    def setup_method(self):
        with patch("insightface.app.FaceAnalysis"):
            self.engine = FaceRecognitionEngine.__new__(FaceRecognitionEngine)

    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert self.engine.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert self.engine.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert self.engine.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_batch_similarity_returns_max(self):
        query = np.array([1.0, 0.0])
        stored = [np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
        result = self.engine.batch_similarity(query, stored)
        assert result == pytest.approx(1.0)

    def test_batch_similarity_empty(self):
        query = np.array([1.0, 0.0])
        assert self.engine.batch_similarity(query, []) == 0.0


# ─── Database tests ───────────────────────────────────────────────────────────

class TestDatabase:
    def setup_method(self):
        self.db = Database(":memory:")
        self.db.init_db()

    def _make_embedding(self):
        v = np.random.randn(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_save_and_load_user(self):
        embs = [self._make_embedding() for _ in range(5)]
        self.db.save_user("u1", "Alice", embs)
        users = self.db.load_all_users()
        assert len(users) == 1
        assert users[0]["name"] == "Alice"
        assert len(users[0]["embeddings"]) == 5

    def test_embedding_values_preserved(self):
        emb = self._make_embedding()
        self.db.save_user("u2", "Bob", [emb])
        users = self.db.load_all_users()
        loaded = users[0]["embeddings"][0]
        np.testing.assert_allclose(emb, loaded, rtol=1e-5)

    def test_upsert_user(self):
        embs = [self._make_embedding()]
        self.db.save_user("u3", "Charlie", embs)
        new_embs = [self._make_embedding(), self._make_embedding()]
        self.db.save_user("u3", "Charlie Updated", new_embs)
        users = self.db.load_all_users()
        assert len(users) == 1
        assert users[0]["name"] == "Charlie Updated"
        assert len(users[0]["embeddings"]) == 2

    def test_delete_user(self):
        self.db.save_user("u4", "Dave", [self._make_embedding()])
        assert self.db.delete_user("u4") is True
        assert self.db.load_all_users() == []

    def test_delete_nonexistent_user(self):
        assert self.db.delete_user("ghost") is False

    def test_user_exists(self):
        self.db.save_user("u5", "Eve", [self._make_embedding()])
        assert self.db.user_exists("u5") is True
        assert self.db.user_exists("nobody") is False

    def test_empty_db_returns_empty_list(self):
        assert self.db.load_all_users() == []
