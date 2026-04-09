import sqlite3
import json
import logging
import os
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "faces.db")


class Database:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        # Keep one connection for in-memory databases so schema/data survive across calls.
        self._memory_conn = None
        if path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row

    def _connect(self):
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    embeddings  TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.info(f"Database ready at {self.path}")

    def save_user(self, user_id: str, name: str, embeddings: list[np.ndarray]):
        emb_list = [e.tolist() for e in embeddings]
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO users (id, name, embeddings, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    embeddings=excluded.embeddings,
                    updated_at=CURRENT_TIMESTAMP
            """, (user_id, name, json.dumps(emb_list)))
            conn.commit()
        logger.info(f"Saved user {user_id} with {len(embeddings)} embeddings")

    def load_all_users(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, name, embeddings FROM users").fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "embeddings": [np.array(e) for e in json.loads(row["embeddings"])]
            }
            for row in rows
        ]

    def delete_user(self, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    def user_exists(self, user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
            return row is not None


database = Database()
