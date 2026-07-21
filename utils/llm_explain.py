import re

import anthropic

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# Word stems banned from the narration, plus close synonyms that smuggle in the
# same meaning. A probe (see hallucination_probe.py) found the model reaching
# for "genuine"/"authentic"/"legitimate" specifically to describe the Reuters
# wire-service dateline -- the exact framing this guardrail exists to prevent,
# just with different words than the ones first flagged.
_BANNED_STEMS = [
    "authentic", "legitimat", "credib", "trustworth", "genuine",
    "reliab", "reputable", "verified", "verification", "fact-check",
    "well-sourced", "objectively true", "real journalism", "real reporting",
]
_BANNED_RE = re.compile("|".join(re.escape(s) for s in _BANNED_STEMS), re.IGNORECASE)


def _violations(text: str) -> list[str]:
    return sorted(set(m.group(0).lower() for m in _BANNED_RE.finditer(text)))


_SYSTEM = (
    "You are an explanation-only assistant for a fake-news classifier. "
    "A separate machine learning model (logistic regression on TF-IDF features) has already "
    "made the real/fake prediction and computed SHAP token attributions. "
    "Your ONLY job is to narrate, in plain English, why the model reached this conclusion, "
    "using strictly the token weights provided.\n\n"
    "IMPORTANT: many high-weight tokens are dataset formatting artifacts (wire-service "
    "datelines like 'WASHINGTON' or '(Reuters', source-name mentions, punctuation habits) that "
    "correlate with the label only because of how the training data was collected and labeled -- "
    "not because they are evidence of real-world credibility, verification, or journalistic "
    "quality. Describe such tokens neutrally, e.g. 'a formatting pattern the model associates "
    "with this label in the training data'. "
    "Never describe a token, phrase, or the article itself as 'authentic', 'legitimate', "
    "'credible', 'trustworthy', 'genuine', 'reliable', 'reputable', 'verified', or "
    "'well-sourced' -- and do not use other wording that means the same thing. These words all "
    "overstate what a statistical correlation means, and are banned regardless of phrasing.\n\n"
    "Only claim that one group of tokens 'outweighs', 'dominates', or 'drives' the prediction "
    "if that is actually consistent with the signs and magnitudes of the weights listed above -- "
    "do not invent a directional story that contradicts the numbers you were given. If the "
    "model's confidence is weak (probability of real between roughly 0.35 and 0.65) or the "
    "listed tokens are mixed/ambiguous, say so plainly instead of manufacturing a confident, "
    "decisive-sounding narrative.\n\n"
    "Do not make your own judgment about whether the article is real or fake. "
    "Do not introduce facts, sources, or claims about the article's accuracy that are not "
    "derivable from the listed tokens. Do not contradict or second-guess the model's prediction. "
    "Keep the explanation to 3-5 sentences."
)

_CORRECTION_TEMPLATE = (
    "Your explanation used these forbidden words or phrases: {violations}. "
    "Every one of them implies real-world credibility/authenticity, which you must never assert -- "
    "these tokens only correlate with the label because of training-data artifacts, not because "
    "the article is actually trustworthy. Rewrite the entire explanation from scratch. Describe "
    "the flagged tokens only as 'a pattern the model associates with this label in the training "
    "data', avoid all of the forbidden words and any synonyms with the same meaning, and keep it "
    "to 3-5 sentences."
)


def narrate_shap(text, label, probability, top_terms, model="claude-haiku-4-5"):
    """Narrate a SHAP explanation in plain English. Does not classify — the
    label/probability/top_terms are already decided by the sklearn model;
    this only describes them."""
    terms_desc = "\n".join(
        f'- "{term}": {"pushes toward REAL" if val > 0 else "pushes toward FAKE"} (weight {val:+.3f})'
        for term, val in top_terms
    )

    user = (
        f"Predicted label: {label}\n"
        f"Model confidence (probability of REAL): {probability:.3f}\n\n"
        f"Top contributing tokens (from SHAP):\n{terms_desc}\n\n"
        f"Article text (for context only, do not re-analyze it yourself):\n{text[:2000]}\n\n"
        "Explain in plain English why the model predicted this label, based only on the token weights above."
    )

    client = _get_client()
    messages = [{"role": "user", "content": user}]

    response = client.messages.create(
        model=model, max_tokens=400, system=_SYSTEM, messages=messages
    )
    rationale = next((b.text for b in response.content if b.type == "text"), "")

    bad = _violations(rationale)
    if bad:
        messages.append({"role": "assistant", "content": rationale})
        messages.append(
            {"role": "user", "content": _CORRECTION_TEMPLATE.format(violations=", ".join(bad))}
        )
        response = client.messages.create(
            model=model, max_tokens=400, system=_SYSTEM, messages=messages
        )
        rationale = next((b.text for b in response.content if b.type == "text"), "")

        bad = _violations(rationale)
        if bad:
            raise RuntimeError(
                f"Claude explanation still used banned credibility-implying language after a "
                f"correction attempt: {', '.join(bad)}. Refusing to serve it."
            )

    return rationale
