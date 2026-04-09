import logging
import os
from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List

from .recognition import FaceRecognitionEngine
from .db import database
from .models import RegisterResponse, RecognizeResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Recognition API",
    version="1.1.0",
    description="ArcFace-based face recognition backend."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = FaceRecognitionEngine()

THRESHOLD  = float(os.getenv("THRESHOLD", "0.45"))
MIN_IMAGES = int(os.getenv("MIN_IMAGES", "3"))


@app.on_event("startup")
def startup():
    database.init_db()
    logger.info(f"Ready. THRESHOLD={THRESHOLD} MIN_IMAGES={MIN_IMAGES}")


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD}


@app.get("/users")
def list_users():
    users = database.load_all_users()
    return {
        "count": len(users),
        "users": [{"id": u["id"], "name": u["name"]} for u in users]
    }


@app.post("/register/{user_id}", response_model=RegisterResponse)
async def register(
    user_id: str,
    name: str = Query(..., min_length=1, max_length=100),
    files: List[UploadFile] = None
):
    """
    Register a user face.
    - user_id: unique identifier (UUID recommended)
    - name: display name
    - files: 3–10 JPEG/PNG face images. More images = better accuracy.
             Images should vary: different lighting, angles, with/without glasses.
             The Android app sends pre-aligned 112x112 crops.
    """
    if not files:
        raise HTTPException(400, "No files uploaded")

    embeddings = []
    failed = 0

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(422, f"{file.filename} is not an image")

        img_bytes = await file.read()
        if len(img_bytes) == 0:
            raise HTTPException(422, f"{file.filename} is empty")

        emb = engine.get_embedding(img_bytes)
        if emb is not None:
            embeddings.append(emb)
        else:
            failed += 1
            logger.warning(f"No face in {file.filename} (user={user_id})")

    if len(embeddings) < MIN_IMAGES:
        raise HTTPException(
            400,
            f"Only {len(embeddings)} valid face(s) detected out of {len(files)} images. "
            f"Need at least {MIN_IMAGES}. {failed} image(s) had no detectable face."
        )

    database.save_user(user_id, name.strip(), embeddings)
    logger.info(f"Registered '{name}' ({user_id}) — {len(embeddings)} embeddings stored")

    return RegisterResponse(
        status="registered",
        user_id=user_id,
        embeddings_stored=len(embeddings)
    )


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(file: UploadFile):
    """
    Identify a face image.
    Returns matched user + confidence if above threshold, otherwise unknown.
    The Android app sends a pre-aligned 112x112 JPEG crop after liveness check.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(422, "File is not an image")

    img_bytes = await file.read()
    if len(img_bytes) == 0:
        raise HTTPException(422, "Empty file")

    query_emb = engine.get_embedding(img_bytes)
    if query_emb is None:
        raise HTTPException(400, "No face detected in image. Ensure the crop is clear and well-lit.")

    users = database.load_all_users()
    if not users:
        return RecognizeResponse(
            matched=False,
            confidence=0.0,
            message="No users registered yet"
        )

    best_score = 0.0
    best_user  = None

    for user in users:
        score = engine.best_match_score(query_emb, user["embeddings"])
        if score > best_score:
            best_score = score
            best_user  = user

    logger.info(
        f"Best match: {best_user['name'] if best_user else 'none'} "
        f"score={best_score:.4f} threshold={THRESHOLD}"
    )

    if best_user and best_score >= THRESHOLD:
        return RecognizeResponse(
            matched=True,
            user_id=best_user["id"],
            name=best_user["name"],
            confidence=round(best_score, 4)
        )

    return RecognizeResponse(
        matched=False,
        confidence=round(best_score, 4),
        message="Face not recognised"
    )


@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    if not database.delete_user(user_id):
        raise HTTPException(404, f"User {user_id} not found")
    return {"status": "deleted", "user_id": user_id}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
