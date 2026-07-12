import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

router = APIRouter(prefix="/data", tags=["dataviz"])


def _read_json(filename: str):
    path = ARTIFACTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found. Run the viz prep script first.")
    with open(path) as f:
        return json.load(f)


@router.get("/class-counts")
def class_counts():
    return _read_json("class_counts.json")


@router.get("/null-rates")
def null_rates():
    return _read_json("null_rates.json")


@router.get("/top-terms")
def top_terms():
    return _read_json("top_terms.json")


@router.get("/weekly-counts")
def weekly_counts():
    path = ARTIFACTS_DIR / "weekly_counts.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="weekly_counts.csv not found. Run the viz prep script first.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@router.get("/pca")
def pca():
    xy_path = ARTIFACTS_DIR / "pca_2d.npy"
    labels_path = ARTIFACTS_DIR / "pca_labels.npy"
    kmeans_path = ARTIFACTS_DIR / "kmeans_labels.npy"
    if not (xy_path.exists() and labels_path.exists() and kmeans_path.exists()):
        raise HTTPException(status_code=404, detail="PCA/KMeans artifacts not found. Run the viz prep script first.")

    xy = np.load(xy_path)
    labels = np.load(labels_path, allow_pickle=True)
    clusters = np.load(kmeans_path)

    return {
        "points": [
            {"x": float(xy[i, 0]), "y": float(xy[i, 1]), "label": str(labels[i]), "cluster": int(clusters[i])}
            for i in range(len(xy))
        ]
    }


@router.get("/topics")
def topics():
    return _read_json("nmf_topics.json")


@router.get("/metrics")
def metrics():
    matches = sorted(ARTIFACTS_DIR.glob("metrics_*.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="No metrics_*.json found. Re-train to generate one.")
    with open(matches[-1]) as f:
        return json.load(f)
