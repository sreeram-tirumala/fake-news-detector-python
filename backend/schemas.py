from typing import List, Literal, Optional

from pydantic import BaseModel


class TopTerm(BaseModel):
    word: str
    shap_value: float


class PredictRequest(BaseModel):
    headline: str = ""
    body: str = ""


class PredictResponse(BaseModel):
    label: Literal["real", "fake"]
    probability: float
    top_terms: List[TopTerm]


class ExplainRequest(BaseModel):
    text: str
    label: Literal["real", "fake"]
    probability: float
    top_terms: List[TopTerm]


class ExplainResponse(BaseModel):
    rationale: str


class UrlPredictRequest(BaseModel):
    url: str


class UrlPredictResponse(BaseModel):
    label: Literal["real", "fake"]
    probability: float
    top_terms: List[TopTerm]
    fetched_title: Optional[str] = None
    fetched_text: str
    fetch_method: Literal["trafilatura", "claude_web_fetch"]


class CorroborateRequest(BaseModel):
    text: str
    url: Optional[str] = None


class CorroborationSourceSchema(BaseModel):
    url: str
    title: str
    stance: Literal["supports", "contradicts", "unrelated"]
    note: str


class CorroborateResponse(BaseModel):
    verdict: Literal["corroborated", "contradicted", "unverifiable", "mixed"]
    rationale: str
    sources: List[CorroborationSourceSchema]
    raw_search_queries: List[str]
