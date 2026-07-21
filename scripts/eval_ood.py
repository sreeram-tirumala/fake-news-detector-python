"""
Out-of-distribution eval harness for the ISOT fake-news classifier.

eval_leakage.py already showed how much of the ~99% reported accuracy is
Reuters-dateline artifact rather than signal. This script asks a different
question: how well does the same pipeline generalize when the *distribution*
of inputs shifts away from what it was trained on -- new time periods, or an
entirely different dataset the model has never seen?

Two modes:

  1. temporal_holdout (always runs, needs only ISOT)
     Train on ISOT articles before --cutoff_date, test on articles from that
     date onward. This is a genuine distribution shift over time (different
     news events, evolving vocabulary) using data already in this repo, so it
     works with no extra setup.

  2. external (opt-in, needs --ood_csv)
     Train on all of ISOT, then evaluate on a completely different fake/real
     news dataset (e.g. LIAR, FakeNewsNet, Kaggle's "Getting Real about Fake
     News"). This is the strongest OOD test but requires a CSV you supply --
     point --ood_csv at it and map its columns with --ood_text_col /
     --ood_label_col / --ood_real_value / --ood_fake_value.

Both modes are reported alongside an in-distribution random-split baseline
(reproducing train_sklearn.py) so the accuracy delta under distribution shift
is explicit.
"""
import argparse, os, json, sys
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)


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


def build_pipeline(cfg):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=cfg["sklearn"]["max_features"],
            ngram_range=tuple(cfg["sklearn"]["ngram_range"]),
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=cfg["sklearn"]["C"],
            penalty=cfg["sklearn"]["penalty"],
            solver=cfg["sklearn"]["solver"],
            max_iter=1000,
        )),
    ])


def evaluate(name, pipe, X_te, y_te, baseline_acc=None):
    y_pred = pipe.predict(X_te)
    try:
        y_proba = pipe.predict_proba(X_te)[:, 1]
        roc_auc = roc_auc_score(y_te, y_proba)
    except ValueError:
        roc_auc = None

    acc = accuracy_score(y_te, y_pred)
    row = {
        "condition": name,
        "n_test": int(len(X_te)),
        "accuracy": acc,
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=[0, 1]).tolist(),
    }
    if baseline_acc is not None:
        row["accuracy_delta_vs_baseline"] = acc - baseline_acc
    return row


def run_baseline(cfg, text, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        text, y, test_size=cfg["sklearn"]["test_size"], random_state=cfg["sklearn"]["random_state"], stratify=y
    )
    pipe = build_pipeline(cfg)
    pipe.fit(X_tr, y_tr)
    return evaluate("baseline_random_split", pipe, X_te, y_te)


def run_temporal_holdout(cfg, df, text, y, cutoff_date, baseline_acc):
    # True.csv's dates carry a trailing space ("December 31, 2017 ") that Fake.csv's
    # don't. pandas' format-inference fast path locks onto the first row's exact
    # format and NaTs every row that doesn't match byte-for-byte, which silently
    # drops ~all fake-labeled dates if not stripped first.
    dates = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce")
    has_date = dates.notna()
    if has_date.sum() == 0:
        print("[WARN] No parseable dates in ISOT data -- skipping temporal_holdout.")
        return None

    cutoff = pd.Timestamp(cutoff_date)
    train_mask = has_date & (dates < cutoff)
    test_mask = has_date & (dates >= cutoff)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(f"[WARN] Cutoff {cutoff_date} leaves an empty train or test split -- skipping temporal_holdout.")
        return None

    pipe = build_pipeline(cfg)
    pipe.fit(text[train_mask], y[train_mask])
    row = evaluate("temporal_holdout", pipe, text[test_mask], y[test_mask], baseline_acc)
    row["cutoff_date"] = cutoff_date
    row["n_train"] = int(train_mask.sum())
    return row


def run_external(cfg, full_text, full_y, ood_csv, text_col, label_col, real_value, fake_value, baseline_acc):
    ood_df = pd.read_csv(ood_csv)
    for col in (text_col, label_col):
        if col not in ood_df.columns:
            raise ValueError(f"Column '{col}' not found in {ood_csv}. Available: {list(ood_df.columns)}")

    ood_df = ood_df.dropna(subset=[text_col, label_col])
    ood_labels = ood_df[label_col].astype(str)
    keep = ood_labels.isin([real_value, fake_value])
    dropped = len(ood_df) - int(keep.sum())
    if dropped:
        print(f"[WARN] Dropping {dropped} external rows whose label isn't '{real_value}' or '{fake_value}'.")
    ood_df, ood_labels = ood_df[keep], ood_labels[keep]

    ood_text = ood_df[text_col].astype(str)
    ood_y = (ood_labels == real_value).astype(int)

    pipe = build_pipeline(cfg)
    pipe.fit(full_text, full_y)
    row = evaluate("external_ood", pipe, ood_text, ood_y, baseline_acc)
    row["source_csv"] = str(ood_csv)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data_dir", default="data/ISOT")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--cutoff_date", default="2017-10-01", help="Temporal holdout split point (train < cutoff, test >= cutoff).")
    ap.add_argument("--ood_csv", default=None, help="Optional path to an external real/fake news CSV for a true OOD test.")
    ap.add_argument("--ood_text_col", default="text")
    ap.add_argument("--ood_label_col", default="label")
    ap.add_argument("--ood_real_value", default="real")
    ap.add_argument("--ood_fake_value", default="fake")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading raw ISOT data...")
    df = load_isot(args.data_dir)
    title = df["title"].fillna("") if "title" in df.columns else ""
    text = (title + " " + df["text"].fillna("")).astype(str) if "title" in df.columns else df["text"].astype(str)
    y = (df["label"] == "real").astype(int)
    print(f"[INFO] Loaded {len(df)} articles: {df['label'].value_counts().to_dict()}")

    results = []

    print("\n===== 1. In-distribution baseline (random split) =====")
    baseline = run_baseline(cfg, text, y)
    print(f"accuracy={baseline['accuracy']:.4f}  f1={baseline['f1']:.4f}")
    results.append(baseline)

    print(f"\n===== 2. Temporal holdout (train < {args.cutoff_date}, test >= {args.cutoff_date}) =====")
    temporal = run_temporal_holdout(cfg, df, text, y, args.cutoff_date, baseline["accuracy"])
    if temporal:
        print(f"accuracy={temporal['accuracy']:.4f}  delta_vs_baseline={temporal['accuracy_delta_vs_baseline']:+.4f}")
        results.append(temporal)

    if args.ood_csv:
        print(f"\n===== 3. External OOD dataset ({args.ood_csv}) =====")
        external = run_external(
            cfg, text, y, args.ood_csv,
            args.ood_text_col, args.ood_label_col, args.ood_real_value, args.ood_fake_value,
            baseline["accuracy"],
        )
        print(f"accuracy={external['accuracy']:.4f}  delta_vs_baseline={external['accuracy_delta_vs_baseline']:+.4f}")
        results.append(external)
    else:
        print(
            "\n===== 3. External OOD dataset =====\n"
            "[SKIPPED] Pass --ood_csv <path> to test against a dataset the model has never seen "
            "(e.g. LIAR, FakeNewsNet, or another Kaggle fake/real news corpus). Map its columns "
            "with --ood_text_col/--ood_label_col/--ood_real_value/--ood_fake_value if they differ "
            "from 'text'/'label'/'real'/'fake'."
        )

    out_path = os.path.join(args.out_dir, "ood_eval.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n===== Summary =====")
    cols = ["condition", "n_test", "accuracy", "accuracy_delta_vs_baseline", "f1", "roc_auc"]
    summary_df = pd.DataFrame(results)
    print(summary_df[[c for c in cols if c in summary_df.columns]].to_string(index=False))
    print(f"\n[INFO] Saved full results -> {out_path}")


if __name__ == "__main__":
    main()
