from pydantic import BaseModel
from typing import Optional


class RegisterResponse(BaseModel):
    status: str
    user_id: str
    embeddings_stored: int


class RecognizeResponse(BaseModel):
    matched: bool
    user_id: Optional[str] = None
    name: Optional[str] = None
    confidence: float
    message: Optional[str] = None
