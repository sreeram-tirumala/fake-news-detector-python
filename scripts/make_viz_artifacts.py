# scripts/make_viz_artifacts.py
import os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans

def top_terms_per_class(texts, labels, n=20, ngram=(1,2)):
    cv = CountVectorizer(max_features=60000, ngram_range=ngram, stop_words='english')
    X = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    # sum counts by class
    mask_real = (labels == "real")
    counts_real = np.asarray(X[mask_real].sum(axis=0)).ravel()
    counts_fake = np.asarray(X[~mask_real].sum(axis=0)).ravel()
    top_real_idx = counts_real.argsort()[::-1][:n]
    top_fake_idx = counts_fake.argsort()[::-1][:n]
    return {
        "real": [{"term": t, "count": int(c)} for t, c in zip(vocab[top_real_idx], counts_real[top_real_idx])],
        "fake": [{"term": t, "count": int(c)} for t, c in zip(vocab[top_fake_idx], counts_fake[top_fake_idx])],
    }

def parse_date_series(df):
    # for ISOT, date often like 'December 31, 2017'; try best-effort parse
    for c in ["date","Date","published","publish_date"]:
        if c in df.columns:
            d = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if d.notna().sum() > 0:
                return d
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train.csv")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--topics", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=6)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # guarantee required cols
    for col in ["text","label"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in {args.csv}")
    texts = (df.get("title","").fillna("") + " " + df["text"].fillna("")).astype(str)
    labels = df["label"].astype(str).str.lower()

    # ---------- Class balance (Data Wrangling) ----------
    cls_counts = labels.value_counts().to_dict()
    with open(os.path.join(args.out_dir, "class_counts.json"), "w") as f:
        json.dump(cls_counts, f, indent=2)

    # ---------- Missingness / Data Imputation preview ----------
    null_rates = df[["title" if "title" in df.columns else "text","text"]].isna().mean().to_dict()
    with open(os.path.join(args.out_dir, "null_rates.json"), "w") as f:
        json.dump(null_rates, f, indent=2)

    # ---------- n-gram frequency per class ----------
    tops = top_terms_per_class(texts, labels, n=25)
    with open(os.path.join(args.out_dir, "top_terms.json"), "w") as f:
        json.dump(tops, f, indent=2)

    # ---------- TF-IDF + PCA (PCA + Cluster Analysis) ----------
    tfidf = TfidfVectorizer(max_features=60000, ngram_range=(1,2), stop_words="english")
    X = tfidf.fit_transform(texts)
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X.toarray()[:8000])  # subsample for speed if huge
    np.save(os.path.join(args.out_dir, "pca_2d.npy"), XY)
    np.save(os.path.join(args.out_dir, "pca_labels.npy"), labels.iloc[:XY.shape[0]].to_numpy())

    # KMeans on same subset
    kmeans = KMeans(n_clusters=args.clusters, n_init="auto", random_state=42).fit(X[:XY.shape[0]])
    np.save(os.path.join(args.out_dir, "kmeans_labels.npy"), kmeans.labels_)

    # ---------- Topic “Factor Analysis” via NMF ----------
    nmf = NMF(n_components=args.topics, random_state=42, init="nndsvda")
    W = nmf.fit_transform(X)
    H = nmf.components_
    words = np.array(tfidf.get_feature_names_out())
    topics = []
    for k, comp in enumerate(H):
        idx = np.argsort(comp)[::-1][:12]
        topics.append({"topic": int(k), "terms": [str(w) for w in words[idx].tolist()]})
    with open(os.path.join(args.out_dir, "nmf_topics.json"), "w") as f:
        json.dump(topics, f, indent=2)

    # ---------- Time series (counts over time) ----------
    dts = parse_date_series(df)
    if dts is not None:
        ts_df = (
            pd.DataFrame({"date": dts, "label": labels})
            .dropna(subset=["date"])
            .assign(week=lambda t: t["date"].dt.to_period("W").dt.start_time)
            .groupby(["week","label"]).size().reset_index(name="count")
        )
        ts_df.to_csv(os.path.join(args.out_dir, "weekly_counts.csv"), index=False)

    print("Visualization artifacts written to:", args.out_dir)

if __name__ == "__main__":
    main()
