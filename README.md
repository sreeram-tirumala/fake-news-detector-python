# AI-Driven Fake News Detector

A fake-news classifier built on the ISOT dataset (TF-IDF + Logistic Regression), with SHAP
explainability, a Claude-powered explanation layer, and a live web-search corroboration agent.
Originally a single-file Streamlit app; now a FastAPI backend + React/TypeScript frontend.

## Architecture

```
backend/            FastAPI app -- serves the model, SHAP, and the two Claude-powered features
  main.py             app entrypoint, CORS
  routers/
    predict.py          /predict, /predict/url, /predict/explain
    corroborate.py      /corroborate
    dataviz.py          /data/*  (class counts, PCA, topics, metrics, ...)
  schemas.py          Pydantic request/response models

frontend/            React + TypeScript (Vite), talks to the backend over HTTP
  src/api/            typed fetch client
  src/components/
    MiniPredictor.tsx   compact predict-and-see-the-chart widget, embedded in the Data Viz tab
    tabs/
      PredictTab.tsx      full 3-tier predict experience (model / SHAP+narration / corroboration)
      DataVizTab.tsx      single dashboard page: MiniPredictor + a dropdown over the 5 EDA views
    dataviz/            one component per EDA view, selected by DataVizTab's dropdown
      DataWranglingSection.tsx, TimeSeriesSection.tsx, PcaClustersSection.tsx,
      TopicsSection.tsx, ModelMetricsSection.tsx

utils/               shared logic, used by the backend
  llm_explain.py        Claude Haiku narrates SHAP token attributions in plain English
  article_fetch.py       URL -> clean article text (trafilatura, falls back to Claude web_fetch)
  rag_agent.py            Claude Sonnet 5 + live web search: corroborates or contradicts an
                          article's claims, independent of the classifier

scripts/             training + evaluation
  prepare_isot.py, train_sklearn.py, train_transformer.py, make_viz_artifacts.py
  eval_leakage.py       quantifies how much of the reported accuracy is dataset-artifact leakage
  eval_ood.py           temporal holdout + optional external-dataset generalization check
```

The old Streamlit app (`app.py`) has been retired now that React has full feature parity across
all 6 original tabs; see git history if you need it.

## A principle this project holds to

**The sklearn model is the only thing that ever says "real" or "fake."** Both Claude-powered
features are explicitly barred from classifying:
- `llm_explain.py` narrates *why* the model decided what it decided (from SHAP token weights) --
  it does not re-decide.
- `rag_agent.py` reports whether independent web sources *corroborate or contradict specific
  claims* in the article -- it does not say "real" or "fake," and its verdict is rendered as a
  visually separate tier so it can never be mistaken for a second vote.

This came out of a real finding: the ISOT dataset's "real" articles almost all carry a
`(Reuters)` wire-service dateline as a formatting artifact, not a genuine signal of
trustworthiness. Early on, the Claude narration layer described that dateline as "credible" --
which would have quietly taught users the wrong lesson about what the model actually detects. The
system prompt now explicitly bans "credible"/"authentic"/"legitimate"/"genuine"/etc. and any
synonym that means the same thing, and a code-level check re-prompts (then hard-fails) the model
if it uses one anyway, since a later probe found it reaching for softer synonyms of the exact
banned words.

## What's been verified so far (2026-07-21)

| Area | Status |
|---|---|
| **Leakage eval** (`scripts/eval_leakage.py`) | Done. Honest accuracy is ~91.56%, not the naively-reported 99.24%, once Reuters-dateline artifacts are stripped and the test split is held out by topic. See `artifacts/leakage_eval.json`. |
| **`scripts/eval_ood.py`** (out-of-distribution eval) | Done. Temporal holdout (train on pre-Oct-2017 ISOT, test on later articles) shows no meaningful degradation (+0.4% vs. random-split baseline) -- consistent with the leakage finding that the model's signal is mostly a stable formatting artifact, not evolving content. Also supports `--ood_csv` to test against a genuinely different dataset (LIAR, FakeNewsNet, etc.) if you have one. See `artifacts/ood_eval.json`. |
| **SHAP + Claude Haiku narration** | Done, tested against real ISOT articles and live-verified in the browser. Anti-hallucination guardrail hardened after a probe caught the model describing the Reuters-dateline artifact as "genuine"/"authentic" reporting -- banned-phrase check now runs in code, not just the prompt. |
| **URL ingestion** (`utils/article_fetch.py`) | Done. trafilatura primary, Claude `web_fetch` fallback for JS-heavy/blocked pages, clean failure path for genuinely inaccessible URLs. |
| **Web corroboration agent** (`utils/rag_agent.py`) | Done. Claude Sonnet 5 + live web search, structured output, anti-hallucination cross-check against actual search results. Verified returning real, distinct, non-fabricated sources. |
| **FastAPI backend** | Done. All endpoints tested directly: `/predict`, `/predict/url`, `/predict/explain`, `/corroborate`, and the 7 `/data/*` endpoints. |
| **React Predict tab** | Done. Full 3-tier UI (model score / SHAP + narration / web corroboration) verified end-to-end in a real browser with live API calls. |
| **React Data Viz tab** | Done. All 5 EDA views (Data Wrangling, Time Series, PCA & Clusters, Topics, Model Metrics) consolidated into a single dashboard page with a dropdown selector, plus an embedded live mini-predictor at the top so you can see a concrete article's real/fake call and SHAP chart in the same view as the aggregate charts. Verified end-to-end in a real browser. |
| **Retiring `app.py`** | Done. Removed now that React has full parity. |

Along the way, found and fixed a real data bug: `True.csv`'s dates carry a trailing space that
`Fake.csv`'s don't, which made pandas' date-format inference silently drop nearly all of one
class's dates once the two are concatenated (it locks a format from the first row and requires an
exact match). This had made the shipped `weekly_counts.csv` artifact show only the fake class.
Fixed in both `scripts/make_viz_artifacts.py` and `scripts/eval_ood.py`, and `prepare_isot.py` now
carries the `date` column through so `data/train.csv` has what it needs to reproduce this from
scratch.

## Is it "live"?

**Yes -- the whole app works end-to-end right now.** Predict (model prediction, SHAP explanation,
Claude narration, live web corroboration) and Data Viz (all 5 EDA views + the embedded
mini-predictor) are both fully wired up in the React/FastAPI app.

## Running it

You'll need a trained model in `artifacts/*.joblib` and the viz artifacts in `artifacts/*.json`
(see Setup below if you don't have them yet), and an Anthropic API key in `.env` (copy
`.env.example` and fill it in) for the two Claude features to work -- Predict still works without
a key, just without narration/corroboration.

```bash
# Terminal 1 -- backend
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 -- frontend
cd frontend
npm run dev
```

Open **http://localhost:5173**.

## Setup (first time)

1. `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. `cd frontend && npm install && cd ..`
3. Drop the ISOT dataset (`True.csv`, `Fake.csv`) into `data/ISOT/`
4. `python scripts/prepare_isot.py --data_dir data/ISOT`
5. `python scripts/train_sklearn.py --config config.yaml`
6. `python scripts/make_viz_artifacts.py --csv data/train.csv` (needed for the Data Viz tab)
7. `cp .env.example .env` and fill in `ANTHROPIC_API_KEY`
8. Run it (see above)
