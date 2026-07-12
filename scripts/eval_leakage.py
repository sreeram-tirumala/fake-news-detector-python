"""
Leakage eval harness for the ISOT fake-news classifier.

The baseline pipeline (train_sklearn.py) hits ~99% accuracy, which is a red
flag: ISOT's "real" articles are Reuters wire copy and its "fake" articles
come from a different scrape/cleaning process, so the two classes carry
strong stylistic fingerprints that have nothing to do with truthfulness:

  - 99.2% of real articles start with a "(Reuters)" dateline; ~0% of fake do
  - 47.4% of real articles contain curly apostrophes (E28099); 0% of fake do
  - 91.2% of fake articles have contractions with the apostrophe stripped
    ("don t" instead of "don't"), vs 42.5% of real
  - the `subject` metadata column splits the classes with zero overlap

This script re-measures the same TF-IDF + Logistic Regression pipeline
under three progressively less "leaky" conditions and reports the accuracy
delta, so we know how much of the reported performance is genuine signal
vs. dataset artifact:

  1. baseline        - original text, random 80/20 split (reproduces train_sklearn.py)
  2. cleaned         - dateline/punctuation/contraction artifacts stripped, random split
  3. source_holdout  - original text, but train/test split by `subject` so the
                       test set is drawn from topic/source groups never seen in training
  4. cleaned+holdout - both interventions combined (the "honest" number)
"""
import argparse, os, re, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

DATELINE_RE = re.compile(r"^[A-Z][A-Za-z.,/'\s]{0,40}\(Reuters\)\s*-\s*")
REUTERS_RE = re.compile(r"\(Reuters\)")

BROKEN_CONTRACTIONS = [
    (r"\b(don|doesn|didn|wasn|weren|isn|aren|wouldn|couldn|shouldn|hasn|haven|hadn|won|can|ain)\s+t\b", r"\1't"),
    (r"\b(he|she|it|that|what|here|there|who|let)\s+s\b", r"\1's"),
    (r"\b(I|you|we|they)\s+re\b", r"\1're"),
    (r"\b(I|you|we|they|it)\s+ll\b", r"\1'll"),
    (r"\b(I|you|we|they)\s+ve\b", r"\1've"),
    (r"\bI\s+d\b", r"I'd"),
]


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_isot(data_dir: str) -> pd.DataFrame:
    df_true = pd.read_csv(os.path.join(data_dir, "True.csv"))
    df_fake = pd.read_csv(os.path.join(data_dir, "Fake.csv"))
    df_true["label"] = "real"
    df_fake["label"] = "fake"
    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    return df


def clean_text(s: str) -> str:
    s = DATELINE_RE.sub("", s)
    s = REUTERS_RE.sub("", s)
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    for pattern, repl in BROKEN_CONTRACTIONS:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    return s


def build_tfidf(cfg):
    return TfidfVectorizer(
        max_features=cfg["sklearn"]["max_features"],
        ngram_range=tuple(cfg["sklearn"]["ngram_range"]),
        stop_words="english",
    )


def build_pipeline(cfg):
    return Pipeline([
        ("tfidf", build_tfidf(cfg)),
        ("clf", LogisticRegression(
            C=cfg["sklearn"]["C"],
            penalty=cfg["sklearn"]["penalty"],
            solver=cfg["sklearn"]["solver"],
            max_iter=1000,
        )),
    ])


def evaluate(cfg, name, X_tr, y_tr, X_te, y_te, out_dir):
    pipe = build_pipeline(cfg)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]

    row = {
        "condition": name,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_proba),
    }

    clf = pipe.named_steps["clf"]
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coefs = clf.coef_[0]
    top_real_idx = coefs.argsort()[::-1][:15]
    top_fake_idx = coefs.argsort()[:15]
    row["top_real_terms"] = vocab[top_real_idx].tolist()
    row["top_fake_terms"] = vocab[top_fake_idx].tolist()

    return row


def random_split(X, y, test_size, seed):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def source_holdout_split(df, text_series):
    """Train/test split by `subject` so test-set sources are unseen in training."""
    test_subjects_real = {"worldnews"}
    test_subjects_fake = {"News", "Middle-east"}

    is_real = df["label"] == "real"
    in_test_subject = np.where(
        is_real, df["subject"].isin(test_subjects_real), df["subject"].isin(test_subjects_fake)
    )

    y = (df["label"] == "real").astype(int)
    X_tr = text_series[~in_test_subject]
    X_te = text_series[in_test_subject]
    y_tr = y[~in_test_subject]
    y_te = y[in_test_subject]
    return X_tr, X_te, y_tr, y_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data_dir", default="data/ISOT")
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading raw ISOT data (need `subject` column, not present in data/train.csv)...")
    df = load_isot(args.data_dir)
    print(f"[INFO] Loaded {len(df)} articles: {df['label'].value_counts().to_dict()}")

    title = df["title"].fillna("")
    text_raw = (title + " " + df["text"].fillna("")).astype(str)
    text_clean = text_raw.map(clean_text)
    y_full = (df["label"] == "real").astype(int)

    results = []

    # 1. Baseline: original text, random split (reproduces train_sklearn.py)
    print("\n===== 1. Baseline (original text, random split) =====")
    X_tr, X_te, y_tr, y_te = random_split(text_raw, y_full, cfg["sklearn"]["test_size"], cfg["sklearn"]["random_state"])
    r = evaluate(cfg, "baseline", X_tr, y_tr, X_te, y_te, args.out_dir)
    print(f"accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}  roc_auc={r['roc_auc']:.4f}")
    print("top 'real' terms:", r["top_real_terms"][:8])
    print("top 'fake' terms:", r["top_fake_terms"][:8])
    results.append(r)

    # 2. Cleaned text, random split
    print("\n===== 2. Cleaned text (dateline/punctuation stripped, random split) =====")
    X_tr, X_te, y_tr, y_te = random_split(text_clean, y_full, cfg["sklearn"]["test_size"], cfg["sklearn"]["random_state"])
    r = evaluate(cfg, "cleaned", X_tr, y_tr, X_te, y_te, args.out_dir)
    print(f"accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}  roc_auc={r['roc_auc']:.4f}")
    print("top 'real' terms:", r["top_real_terms"][:8])
    print("top 'fake' terms:", r["top_fake_terms"][:8])
    results.append(r)

    # 3. Source holdout: original text, split by subject
    print("\n===== 3. Source holdout (original text, unseen subjects in test) =====")
    X_tr, X_te, y_tr, y_te = source_holdout_split(df, text_raw)
    r = evaluate(cfg, "source_holdout", X_tr, y_tr, X_te, y_te, args.out_dir)
    print(f"accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}  roc_auc={r['roc_auc']:.4f}")
    results.append(r)

    # 4. Cleaned + source holdout (the "honest" number)
    print("\n===== 4. Cleaned + source holdout (both interventions combined) =====")
    X_tr, X_te, y_tr, y_te = source_holdout_split(df, text_clean)
    r = evaluate(cfg, "cleaned_source_holdout", X_tr, y_tr, X_te, y_te, args.out_dir)
    print(f"accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}  roc_auc={r['roc_auc']:.4f}")
    results.append(r)

    out_path = os.path.join(args.out_dir, "leakage_eval.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n===== Summary =====")
    summary_df = pd.DataFrame(results)[["condition", "n_train", "n_test", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    print(summary_df.to_string(index=False))
    print(f"\n[INFO] Saved full results (incl. top terms) -> {out_path}")


if __name__ == "__main__":
    main()
