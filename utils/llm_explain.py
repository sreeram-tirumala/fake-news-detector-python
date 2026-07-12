import anthropic

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def narrate_shap(text, label, probability, top_terms, model="claude-haiku-4-5"):
    """Narrate a SHAP explanation in plain English. Does not classify — the
    label/probability/top_terms are already decided by the sklearn model;
    this only describes them."""
    terms_desc = "\n".join(
        f'- "{term}": {"pushes toward REAL" if val > 0 else "pushes toward FAKE"} (weight {val:+.3f})'
        for term, val in top_terms
    )

    system = (
        "You are an explanation-only assistant for a fake-news classifier. "
        "A separate machine learning model (logistic regression on TF-IDF features) has already "
        "made the real/fake prediction and computed SHAP token attributions. "
        "Your ONLY job is to narrate, in plain English, why the model reached this conclusion, "
        "using strictly the token weights provided. "
        "IMPORTANT: many high-weight tokens are dataset formatting artifacts (wire-service "
        "datelines like 'WASHINGTON' or '(Reuters', source-name mentions, punctuation habits) that "
        "correlate with the label only because of how the training data was collected and labeled — "
        "not because they are evidence of real-world credibility, verification, or journalistic "
        "quality. Describe such tokens neutrally, e.g. 'a formatting pattern the model associates "
        "with this label in the training data'. Never describe a token as 'authentic', 'legitimate', "
        "'credible', 'trustworthy', or 'genuine' reporting — that overstates what a statistical "
        "correlation means. "
        "Do not make your own judgment about whether the article is real or fake. "
        "Do not introduce facts, sources, or claims about the article's accuracy that are not "
        "derivable from the listed tokens. Do not contradict or second-guess the model's prediction. "
        "Keep the explanation to 3-5 sentences."
    )

    user = (
        f"Predicted label: {label}\n"
        f"Model confidence (probability of REAL): {probability:.3f}\n\n"
        f"Top contributing tokens (from SHAP):\n{terms_desc}\n\n"
        f"Article text (for context only, do not re-analyze it yourself):\n{text[:2000]}\n\n"
        "Explain in plain English why the model predicted this label, based only on the token weights above."
    )

    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")
