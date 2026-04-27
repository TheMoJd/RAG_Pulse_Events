"""Schémas Pydantic pour l'API."""
from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question de l'utilisateur")


class SourceItem(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    daterange: Optional[str] = None
    location_name: Optional[str] = None
    image: Optional[str] = None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class RebuildRequest(BaseModel):
    use_snapshot: bool = Field(
        default=False,
        description="Si True, utilise le dernier snapshot local au lieu de fetch l'API.",
    )


class RebuildResponse(BaseModel):
    rebuilt: bool
    n_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    index_size: int
    model: str
