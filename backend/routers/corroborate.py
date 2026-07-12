from fastapi import APIRouter, HTTPException

from backend.schemas import CorroborateRequest, CorroborateResponse, CorroborationSourceSchema
from utils.rag_agent import corroborate_claims

router = APIRouter(prefix="/corroborate", tags=["corroborate"])


@router.post("", response_model=CorroborateResponse)
def corroborate(req: CorroborateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Provide article text to corroborate.")

    try:
        result = corroborate_claims(req.text, article_url=req.url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Corroboration agent unavailable: {e}")

    return CorroborateResponse(
        verdict=result.verdict,
        rationale=result.rationale,
        sources=[
            CorroborationSourceSchema(url=s.url, title=s.title, stance=s.stance, note=s.note)
            for s in result.sources
        ],
        raw_search_queries=result.raw_search_queries,
    )
