import logging

from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from typing import Optional

from question_similarity.runtime.engine import Engine
from question_similarity.runtime.models import SimilarityRequest, SimilarityResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Question Similarity API",
    description="API for computing question similarity using sentence transformers",
    version="0.1.0",
)

logger.info("Loading model engine...")
engine: Optional[SentenceTransformer] = Engine()
logger.info("Model engine loaded.")


@app.post("/similarity", response_model=SimilarityResponse)
def compute_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """
    Compute similarity between two sentences.

    Args:
        request: Request containing two sentences

    Returns:
        Similarity prediction in the form of a SimilarityResponse
    """
    if engine is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. Check server startup logs."
        )

    logger.info(
        "Predicting similarity for: '%s' vs '%s'", request.sentence1, request.sentence2
    )
    is_similar, similarity = engine.predict(request.sentence1, request.sentence2)
    logger.info("Result: is_similar=%s, similarity=%.4f", is_similar, similarity)

    return SimilarityResponse(
        is_similar=is_similar,
        sentence1=request.sentence1,
        sentence2=request.sentence2,
        similarity=similarity,
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": engine is not None,
        "similarity_threshold": engine.final_similarity_accuracy if engine else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
