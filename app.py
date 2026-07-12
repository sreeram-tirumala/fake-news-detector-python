import streamlit as st
import shap, os, re
from utils.io import load_artifact, newest_artifact

st.set_page_config(page_title='AI Fake News Detector', page_icon='📰', layout='wide')
st.title('📰 AI-Driven Fake News Detector')


import json, numpy as np, pandas as pd, altair as alt
import matplotlib.pyplot as plt
from utils.io import load_artifact, newest_artifact
import os, glob

default_model = newest_artifact("artifacts/*.joblib")
model_path = st.text_input("Model path (.joblib sklearn)", value=default_model or "")
model = load_artifact(model_path) if model_path and os.path.exists(model_path) else None

tabs = st.tabs(["Predict", "Data Wrangling", "Time Series", "PCA & Clusters", "Topics (NMF)", "Model Metrics"])

# ---------- PREDICT ----------
with tabs[0]:
    with st.expander("How to use", expanded=False):
        #st.write("1) Train: `python -m scripts.train_sklearn --config config.yaml`")
        #st.write("2) Prepare viz: `python -m scripts.make_viz_artifacts --csv data/train.csv`")
        st.write("3) Paste text and predict.")
    headline = st.text_input("Headline (optional)")
    body = st.text_area("Article text", height=220)
    explain_with_claude = st.checkbox(
        "Explain with Claude Haiku (plain-English rationale from SHAP tokens)", value=False
    )
    col_pred, col_expl = st.columns([1,1])
    if st.button("Predict") and model is not None:
        text = (headline + " " + body).strip()
        if not text:
            st.warning("Please paste some text.")
        else:
            proba = model.predict_proba([text])[0,1]
            label = "real" if proba >= 0.5 else "fake"
            with col_pred:
                st.metric("Predicted label", label)
                st.write(f"**Probability (real):** {proba:.3f}")

            top_terms = []
            try:
                import shap
                clf = model.named_steps['clf']
                tfidf = model.named_steps['tfidf']

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

                top_terms = sorted(cleaned, key=lambda t: abs(t[1]), reverse=True)[:10]

                if top_terms:
                    st.subheader("Top contributing words")
                    st.caption(
                        "Words with the largest SHAP contribution to this prediction. "
                        "Positive pushes toward REAL, negative pushes toward FAKE."
                    )
                    terms_df = pd.DataFrame(top_terms, columns=["word", "shap_value"])
                    terms_df["direction"] = terms_df["shap_value"].apply(
                        lambda v: "toward real" if v > 0 else "toward fake"
                    )
                    chart = alt.Chart(terms_df).mark_bar().encode(
                        x=alt.X("shap_value:Q", title="SHAP contribution"),
                        y=alt.Y("word:N", sort="-x", title=None),
                        color=alt.Color(
                            "direction:N",
                            scale=alt.Scale(domain=["toward real", "toward fake"], range=["#2a9d8f", "#e76f51"]),
                            legend=alt.Legend(title=None),
                        ),
                        tooltip=["word", "shap_value"],
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No clean SHAP tokens to display for this text.")
            except Exception:
                st.info("SHAP explanation unavailable; predictions still work.")

            if explain_with_claude:
                if not top_terms:
                    st.info("No SHAP tokens available to narrate.")
                else:
                    try:
                        from utils.llm_explain import narrate_shap
                        st.subheader("Plain-English rationale (Claude Haiku)")
                        with st.spinner("Asking Claude to narrate the SHAP evidence..."):
                            rationale = narrate_shap(text, label, proba, top_terms)
                        st.write(rationale)
                    except Exception as e:
                        st.info(f"Claude explanation unavailable: {e}")
    else:
        st.info("Train a model first to enable predictions.")

# ---------- DATA WRANGLING ----------
with tabs[1]:
    st.subheader("Class Balance & Missingness")
    try:
        counts = json.load(open("artifacts/class_counts.json"))
        dfc = pd.DataFrame({"label": list(counts.keys()), "count": list(counts.values())})
        st.bar_chart(dfc.set_index("label"))
    except Exception:
        st.info("Run: `python -m scripts.make_viz_artifacts --csv data/train.csv`")

    try:
        nulls = json.load(open("artifacts/null_rates.json"))
        dfn = pd.DataFrame({"column": list(nulls.keys()), "null_rate": list(nulls.values())})
        chart = alt.Chart(dfn).mark_bar().encode(x="column", y="null_rate")
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        pass

    st.subheader("Top n-grams by class")
    try:
        tt = json.load(open("artifacts/top_terms.json"))
        for lbl in ["real","fake"]:
            st.markdown(f"**{lbl.title()}**")
            st.dataframe(pd.DataFrame(tt[lbl]))
    except Exception:
        st.info("Top terms not available yet.")

# ---------- TIME SERIES ----------
with tabs[2]:
    st.subheader("Weekly article counts by class")
    if os.path.exists("artifacts/weekly_counts.csv"):
        tdf = pd.read_csv("artifacts/weekly_counts.csv")
        chart = alt.Chart(tdf).mark_line(point=True).encode(
            x="week:T", y="count:Q", color="label:N"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No parseable dates found in your dataset.")

# ---------- PCA & CLUSTERS ----------
with tabs[3]:
    st.subheader("PCA (2D) with K-Means")
    try:
        XY = np.load("artifacts/pca_2d.npy")
        y_lbl = np.load("artifacts/pca_labels.npy", allow_pickle=True)
        km   = np.load("artifacts/kmeans_labels.npy")
        dfp = pd.DataFrame({"x": XY[:,0], "y": XY[:,1], "label": y_lbl, "cluster": km.astype(int)})
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Colored by ground-truth label")
            st.altair_chart(alt.Chart(dfp).mark_circle(size=35).encode(x="x", y="y", color="label"), use_container_width=True)
        with c2:
            st.caption("Colored by KMeans cluster")
            st.altair_chart(alt.Chart(dfp).mark_circle(size=35).encode(x="x", y="y", color="cluster:N"), use_container_width=True)
    except Exception:
        st.info("Run the viz prep script to generate PCA & KMeans artifacts.")

# ---------- TOPICS ----------
with tabs[4]:
    st.subheader("NMF Topics (as Factor-Analysis style)")
    try:
        topics = json.load(open("artifacts/nmf_topics.json"))
        for t in topics:
            st.write(f"**Topic {t['topic']}** — " + ", ".join(t["terms"]))
    except Exception:
        st.info("Run the viz prep script to generate topics.")

# ---------- MODEL METRICS ----------
with tabs[5]:
    st.subheader("Classification Report & Curves")
    # find newest metrics file
    mets = sorted(glob.glob("artifacts/metrics_*.json"))
    if mets:
        with open(mets[-1], "r") as fh:
            M = json.load(fh)
        st.write("**Confusion Matrix**")
        st.table(pd.DataFrame(M["confusion_matrix"], index=["fake","real"], columns=["fake","real"]))
        st.write("**Per-class Metrics**")
        st.json(M["report"])

        st.write("**ROC Curve**")
        fig, ax = plt.subplots()
        fpr = M["curves"]["roc"]["fpr"]; tpr = M["curves"]["roc"]["tpr"]; auc = M["curves"]["roc"]["auc"]
        ax.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
        ax.plot([0,1],[0,1],'--', alpha=.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right")
        st.pyplot(fig)

        st.write("**Precision–Recall Curve**")
        fig2, ax2 = plt.subplots()
        prec = M["curves"]["pr"]["precision"]; rec = M["curves"]["pr"]["recall"]
        ax2.plot(rec, prec)
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        st.pyplot(fig2)
    else:
        st.info("Re-train to generate `metrics_*.json`.")
