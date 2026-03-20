from pydantic import BaseModel


class SimilarityRequest(BaseModel):
    """Request body for the similarity endpoint."""

    sentence1: str
    sentence2: str


class SimilarityResponse(BaseModel):
    """Response body for the similarity endpoint."""

    is_similar: bool
    sentence1: str
    sentence2: str
    similarity: float
