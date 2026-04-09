"""
threshold_test.py

Run this after registering real users to find the optimal similarity threshold.
Usage: python threshold_test.py

It prints cross-user similarity scores so you can set THRESHOLD just above the
highest false-positive score you observe.
"""

import itertools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.recognition import FaceRecognitionEngine
from app.db import database

def run():
    database.init_db()
    users = database.load_all_users()

    if len(users) < 2:
        print("Need at least 2 registered users to test cross-user scores.")
        return

    engine = FaceRecognitionEngine()
    print(f"\nTesting {len(users)} users...\n")

    max_cross_score = 0.0
    min_same_score = 1.0

    # Cross-user scores (should be LOW — these are impostor pairs)
    print("=== Cross-user scores (target: all below your threshold) ===")
    for u1, u2 in itertools.combinations(users, 2):
        for e1 in u1["embeddings"]:
            for e2 in u2["embeddings"]:
                score = engine.cosine_similarity(e1, e2)
                if score > 0.35:  # only print concerning scores
                    print(f"  ⚠ {u1['name']} vs {u2['name']}: {score:.4f}")
                max_cross_score = max(max_cross_score, score)

    # Same-user scores (should be HIGH — these are genuine pairs)
    print("\n=== Same-user scores (target: all above your threshold) ===")
    for user in users:
        embs = user["embeddings"]
        if len(embs) < 2:
            continue
        for e1, e2 in itertools.combinations(embs, 2):
            score = engine.cosine_similarity(e1, e2)
            min_same_score = min(min_same_score, score)
            if score < 0.50:
                print(f"  ⚠ {user['name']} (same person): {score:.4f} — low! check image quality")

    print(f"\n{'='*50}")
    print(f"Max cross-user score : {max_cross_score:.4f}  ← set THRESHOLD above this")
    print(f"Min same-user score  : {min_same_score:.4f}  ← set THRESHOLD below this")
    suggested = (max_cross_score + min_same_score) / 2
    print(f"Suggested THRESHOLD  : {suggested:.4f}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    run()
