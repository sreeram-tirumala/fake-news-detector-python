import re
from pathlib import Path

import numpy as np
import shap
from fastapi import APIRouter, HTTPException

from backend.schemas import (
    ExplainRequest,
    ExplainResponse,
    PredictRequest,
    PredictResponse,
    TopTerm,
    UrlPredictRequest,
    UrlPredictResponse,
)
from utils.article_fetch import fetch_article
from utils.io import load_artifact, newest_artifact
from utils.llm_explain import narrate_shap

router = APIRouter(prefix="/predict", tags=["predict"])

ARTIFACTS_GLOB = str(Path(__file__).resolve().parents[2] / "artifacts" / "*.joblib")

_model = None


def get_model():
    global _model
    if _model is None:
        model_path = newest_artifact(ARTIFACTS_GLOB)
        if not model_path:
            raise HTTPException(status_code=503, detail="No trained model found in artifacts/. Train one first.")
        _model = load_artifact(model_path)
    return _model


def extract_top_terms(model, text: str, n: int = 10):
    clf = model.named_steps["clf"]
    tfidf = model.named_steps["tfidf"]

    def predict_fn(texts):
        return clf.predict_proba(tfidf.transform(texts))[:, 1]

    explainer = shap.Explainer(predict_fn, shap.maskers.Text(r"\w+"))
    sv = explainer([text])

    values = np.asarray(sv[0].values)
    if values.ndim > 1:
        values = values[:, -1]
    tokens = list(sv[0].data)

    # SHAP's text masker leaves punctuation-fused fragments (e.g. '") - The"',
    # bare '","'). Strip non-word characters and drop anything with no letters.
    cleaned = []
    for tok, val in zip(tokens, values):
        t = re.sub(r"[^\w\s'-]", " ", str(tok))
        t = re.sub(r"\s+", " ", t).strip()
        t = t.strip(" -'")
        if t and any(c.isalpha() for c in t):
            cleaned.append((t, float(val)))

    ranked = sorted(cleaned, key=lambda t: abs(t[1]), reverse=True)[:n]
    return [TopTerm(word=w, shap_value=v) for w, v in ranked]


def _predict_from_text(model, text: str):
    proba = float(model.predict_proba([text])[0, 1])
    label = "real" if proba >= 0.5 else "fake"
    try:
        top_terms = extract_top_terms(model, text)
    except Exception:
        top_terms = []
    return label, proba, top_terms


@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model()
    text = (req.headline + " " + req.body).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Provide headline and/or body text.")

    label, proba, top_terms = _predict_from_text(model, text)
    return PredictResponse(label=label, probability=proba, top_terms=top_terms)


@router.post("/url", response_model=UrlPredictResponse)
def predict_from_url(req: UrlPredictRequest):
    fetch_result = fetch_article(req.url)
    if fetch_result.method == "failed" or not fetch_result.text:
        raise HTTPException(status_code=422, detail=f"Could not extract article text: {fetch_result.error}")

    model = get_model()
    text = ((fetch_result.title or "") + " " + fetch_result.text).strip()
    label, proba, top_terms = _predict_from_text(model, text)

    return UrlPredictResponse(
        label=label,
        probability=proba,
        top_terms=top_terms,
        fetched_title=fetch_result.title,
        fetched_text=fetch_result.text,
        fetch_method=fetch_result.method,
    )


@router.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    if not req.top_terms:
        raise HTTPException(status_code=400, detail="No SHAP top_terms provided to explain.")
    try:
        rationale = narrate_shap(
            req.text,
            req.label,
            req.probability,
            [(t.word, t.shap_value) for t in req.top_terms],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Claude explanation unavailable: {e}")
    return ExplainResponse(rationale=rationale)
