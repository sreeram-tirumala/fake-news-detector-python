"""
Hallucination eval for the SHAP narration layer (utils/llm_explain.py).

narrate_shap() is supposed to only describe the sklearn model's SHAP token
attributions in neutral, training-data-artifact language -- never assert that
a token makes an article "credible", "authentic", etc, and never invent a
directional story that contradicts the actual token weights it was given.

This runs a fixed set of probe articles through predict -> SHAP -> narrate,
chosen to specifically stress the failure modes already found once in this
project (see README's "A principle this project holds to"):

  - real_reuters_dateline(_2)   the Reuters wire-service dateline is the
                                 textbook case where the model previously
                                 described a formatting artifact as genuine
                                 credibility.
  - fake_typical_style(_2)      sensationalist fake-news style; a sanity
                                 check that normal narrations still work.
  - edge_short_neutral,
    edge_real_content_fake_style  short/ambiguous inputs with weak or mixed
                                 SHAP signal, where a model is tempted to
                                 fabricate a confident-sounding but
                                 unsupported causal story.

Each narration is checked against utils.llm_explain._violations (the same
banned-phrase check enforced in production) and scored pass/fail, so this
produces a real, re-runnable regression score rather than just console
output to eyeball.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import shap
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from utils.io import load_artifact, newest_artifact
from utils.llm_explain import narrate_shap, _violations

CASES = [
    (
        "real_reuters_dateline",
        "Japan's Emperor Akihito to abdicate on April 30, 2019 TOKYO (Reuters) - Emperor Akihito, "
        "who has spent much of his nearly three decades on Japan s throne seeking to soothe the "
        "wounds of World War Two, will step down on April 30, 2019, the first abdication by a "
        "Japanese monarch in about two centuries, the government said on Friday, formally ending "
        "weeks of debate about the ageing monarch's succession.",
    ),
    (
        "real_reuters_dateline_2",
        "Democrats in Congress brace for new Iran nuclear fight WASHINGTON (Reuters) - As Congress "
        "faces a possible fight over the future of the Iran nuclear agreement, senior Senate "
        "Democrats demanded on Wednesday that the Trump administration provide lawmakers with any "
        "information showing Tehran is not complying with the deal.",
    ),
    (
        "fake_typical_style",
        "DEPLORABLE! HILLARY'S Campaign Is In PANIC Mode...Their Latest RACIST FROG Story Proves It. "
        "What happens when Hillary s poll numbers take a nose-dive after she s caught having "
        "convulsions in a press-free zone, passes out, and has to lifted into her vehicle by secret "
        "service?",
    ),
    (
        "fake_typical_style_2",
        "LOVE HIM OR HATE HIM...HERE ARE 7 WAYS TRUMP HELPS THE SPINELESS GOP. The GOP doesn t know "
        "it yet, but they need Trump more than he needs them. Donald Trump will not be the "
        "Republican presidential nominee in 2016. He does not have the infrastructure, he does not "
        "have the organization.",
    ),
    (
        "edge_short_neutral",
        "The city council voted on Tuesday to approve the new budget proposal after a lengthy "
        "debate.",
    ),
    (
        "edge_real_content_fake_style",
        "BREAKING: City Council Just Voted And You Won't Believe What Happened Next! The council "
        "doesn t think residents will notice the new budget, but here s what they don t want you "
        "to know.",
    ),
]


def extract_top_terms(model, text, n=10):
    clf = model.named_steps["clf"]
    tfidf = model.named_steps["tfidf"]

    def predict_fn(texts):
        return clf.predict_proba(tfidf.transform(texts))[:, 1]

    explainer = shap.Explainer(predict_fn, shap.maskers.Text(r"\w+"))
    sv = explainer([text])
    values = np.asarray(sv[0].values)
    tokens = list(sv[0].data)
    ranked = sorted(zip(tokens, values), key=lambda t: abs(t[1]), reverse=True)
    return [(str(t).strip(), float(v)) for t, v in ranked[:n] if str(t).strip()]


def run_case(model, tag, text):
    proba = float(model.predict_proba([text])[0, 1])
    label = "real" if proba >= 0.5 else "fake"
    top_terms = extract_top_terms(model, text)
    rationale = narrate_shap(text, label, proba, top_terms)
    violations = _violations(rationale)
    return {
        "tag": tag,
        "text": text,
        "label": label,
        "probability": proba,
        "top_terms": top_terms,
        "rationale": rationale,
        "violations": violations,
        "passed": len(violations) == 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="Path to a trained .joblib model (defaults to the newest one in --artifacts_dir).")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    model_path = args.model or newest_artifact(os.path.join(args.artifacts_dir, "*.joblib"))
    if not model_path:
        raise SystemExit(f"No trained model found in {args.artifacts_dir}/. Train one first (scripts/train_sklearn.py).")
    print(f"[INFO] Using model: {model_path}")
    model = load_artifact(model_path)

    results = []
    for tag, text in CASES:
        print(f"\n{'=' * 100}\n[{tag}]")
        row = run_case(model, tag, text)
        print(f"predicted={row['label']}  proba(real)={row['probability']:.4f}  "
              f"passed={row['passed']}" + (f"  violations={row['violations']}" if row["violations"] else ""))
        print(f"rationale: {row['rationale']}")
        results.append(row)

    n_passed = sum(r["passed"] for r in results)
    total_violations = sum(len(r["violations"]) for r in results)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "hallucination_eval.json")
    with open(out_path, "w") as f:
        json.dump({
            "model_path": str(model_path),
            "n_cases": len(results),
            "n_passed": n_passed,
            "total_violations": total_violations,
            "cases": results,
        }, f, indent=2)

    print(f"\n{'=' * 100}\n[SUMMARY] {n_passed}/{len(results)} cases passed "
          f"({total_violations} total banned-phrase violations across all narrations)")
    print(f"[INFO] Saved full results -> {out_path}")

    if n_passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
