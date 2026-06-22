# AI-Driven Fake News Detector (Python-only)

This starter lets you build a Fake News Detector web app with Streamlit, plus training scripts (scikit-learn baseline; optional transformer) and SHAP explainability.

## 1) Pick a dataset (drop it into data/)
- ISOT Fake News Dataset (True.csv, Fake.csv)
- LIAR (short political claims)
- Constraint/CoAID (COVID-19 misinformation)

## 2) Quick start
1) Create a venv and install: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2) Prepare data (ISOT): `python scripts/prepare_isot.py --data_dir data`
3) Train baseline: `python scripts/train_sklearn.py --config config.yaml`
4) (Optional) Train transformer: `python scripts/train_transformer.py --config config.yaml`
5) Launch app: `streamlit run app.py`

## Structure
app.py, config.yaml, requirements.txt, data/, artifacts/, scripts/, utils/
