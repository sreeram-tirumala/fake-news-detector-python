import argparse, os, yaml, time
import pandas as pd, numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--out_dir', default='artifacts')
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg['dataset']['train_csv'])
    text_col = cfg['dataset']['text_col']
    title_col = cfg['dataset'].get('title_col') or ''
    if title_col and title_col in df.columns:
        texts = (df[title_col].fillna('') + ' ' + df[text_col].fillna('')).astype(str).tolist()
    else:
        texts = df[text_col].astype(str).tolist()
    labels = (df['label'] == cfg['dataset']['positive_label']).astype(int).tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    tok = AutoTokenizer.from_pretrained(cfg['transformer']['model_name'])

    def tokenize(batch):
        return tok(batch['text'], truncation=True, padding='max_length', max_length=cfg['transformer']['max_length'])

    train_ds = Dataset.from_dict({'text': X_tr, 'label': y_tr}).map(tokenize, batched=True)
    test_ds  = Dataset.from_dict({'text': X_te, 'label': y_te}).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(cfg['transformer']['model_name'], num_labels=2)

    args_hf = TrainingArguments(
        output_dir='artifacts/hf_runs',
        learning_rate=cfg['transformer']['lr'],
        per_device_train_batch_size=cfg['transformer']['batch_size'],
        per_device_eval_batch_size=cfg['transformer']['batch_size'],
        num_train_epochs=cfg['transformer']['epochs'],
        weight_decay=0.01,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss'
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {'accuracy': accuracy_score(labels, preds), 'f1': f1_score(labels, preds)}

    trainer = Trainer(model=model, args=args_hf, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics)
    trainer.train()

    ts = int(time.time())
    out_path = os.path.join(args.out_dir, f"hf_{cfg['transformer']['model_name'].replace('/','_')}_{ts}")
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    print(f'Saved transformer model -> {out_path}')

if __name__ == '__main__':
    main()
