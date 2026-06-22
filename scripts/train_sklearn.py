import argparse, os, yaml, time, json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

from utils.io import save_artifact


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_tfidf(cfg):
    return TfidfVectorizer(
        max_features=cfg["sklearn"]["max_features"],
        ngram_range=tuple(cfg["sklearn"]["ngram_range"]),
        stop_words="english"
    )


def evaluate_row(name, y_true, y_pred, y_proba=None):
    row = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": None,
    }
    if y_proba is not None:
        try:
            row["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            row["roc_auc"] = None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    # ========================= Load data =========================
    data = pd.read_csv(cfg["dataset"]["train_csv"])

    text_col = cfg["dataset"]["text_col"]
    title_col = cfg["dataset"].get("title_col") or ""
    label_col = cfg["dataset"]["label_col"]
    pos_label = cfg["dataset"]["positive_label"]

    if title_col and title_col in data.columns:
        X_full = (data[title_col].fillna("") + " " + data[text_col].fillna("")).astype(str)
    else:
        X_full = data[text_col].astype(str)

    y_full = (data[label_col] == pos_label).astype(int)

    # ==================== Unsupervised: PCA + KMeans ====================
    print("\n===== Unsupervised: TF-IDF + PCA + KMeans =====")

    tfidf_unsup = build_tfidf(cfg)
    X_tfidf_full = tfidf_unsup.fit_transform(X_full)

    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf_full)

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_tfidf_full)

    np.save(os.path.join(args.out_dir, "pca_2d.npy"), X_2d)
    np.save(os.path.join(args.out_dir, "pca_labels.npy"), data[label_col].values)
    np.save(os.path.join(args.out_dir, "kmeans_labels.npy"), cluster_labels)

    print("[INFO] Saved PCA & KMeans artifacts: pca_2d.npy, pca_labels.npy, kmeans_labels.npy")

    # ==================== Train/Test Split ====================
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full,
        test_size=cfg["sklearn"]["test_size"],
        random_state=cfg["sklearn"]["random_state"],
        stratify=y_full
    )

    results = []

    # ==================== Model 1: Logistic Regression ====================
    print("\n===== Model 1: Logistic Regression (TF-IDF) =====")

    logreg_pipe = Pipeline([
        ("tfidf", build_tfidf(cfg)),
        ("clf", LogisticRegression(
            C=cfg["sklearn"]["C"],
            penalty=cfg["sklearn"]["penalty"],
            solver=cfg["sklearn"]["solver"],
            max_iter=1000
        ))
    ])

    logreg_pipe.fit(X_tr, y_tr)
    y_pred_lr = logreg_pipe.predict(X_te)
    y_proba_lr = logreg_pipe.predict_proba(X_te)[:, 1]

    print(classification_report(y_te, y_pred_lr, target_names=["fake", "real"]))
    try:
        print(f"AUC: {roc_auc_score(y_te, y_proba_lr):.4f}")
    except Exception:
        pass

    cm = confusion_matrix(y_te, y_pred_lr).tolist()
    fpr, tpr, _ = roc_curve(y_te, y_proba_lr)
    prec, rec, _ = precision_recall_curve(y_te, y_proba_lr)
    curves = {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
    }

    ts = int(time.time())
    lr_model_path = os.path.join(args.out_dir, f"sklearn_logreg_{ts}.joblib")
    save_artifact(logreg_pipe, lr_model_path)

    with open(os.path.join(args.out_dir, f"metrics_{ts}.json"), "w") as fh:
        json.dump({
            "report": classification_report(y_te, y_pred_lr, output_dict=True),
            "confusion_matrix": cm,
            "curves": curves,
        }, fh, indent=2)

    print(f"[INFO] Saved Logistic Regression model -> {lr_model_path}")
    results.append(evaluate_row("LogReg TF-IDF", y_te, y_pred_lr, y_proba_lr))

    # ==================== Model 2: Linear SVM ====================
    print("\n===== Model 2: Linear SVM (TF-IDF) =====")

    svm_pipe = Pipeline([
        ("tfidf", build_tfidf(cfg)),
        ("clf", LinearSVC())
    ])

    svm_pipe.fit(X_tr, y_tr)
    y_pred_svm = svm_pipe.predict(X_te)

    print(classification_report(y_te, y_pred_svm, target_names=["fake", "real"]))
    results.append(evaluate_row("Linear SVM TF-IDF", y_te, y_pred_svm))

    # ==================== Model 3: Random Forest ====================
    print("\n===== Model 3: Random Forest (TF-IDF) =====")

    rf_pipe = Pipeline([
        ("tfidf", build_tfidf(cfg)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        ))
    ])

    rf_pipe.fit(X_tr, y_tr)
    y_pred_rf = rf_pipe.predict(X_te)

    print(classification_report(y_te, y_pred_rf, target_names=["fake", "real"]))
    results.append(evaluate_row("Random Forest TF-IDF", y_te, y_pred_rf))

    # ==================== Save comparison ====================
    results_df = pd.DataFrame(results)
    comp_csv = os.path.join(args.out_dir, "model_comparison_part2.csv")
    comp_json = os.path.join(args.out_dir, "model_comparison_part2.json")

    results_df.to_csv(comp_csv, index=False)
    with open(comp_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== Model Comparison (Part 2) =====")
    print(results_df)
    print(f"[INFO] Saved comparison to {comp_csv}")
    print("[DONE] Training + evaluation complete.")


if __name__ == "__main__":
    main()
