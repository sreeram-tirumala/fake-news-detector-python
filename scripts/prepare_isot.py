import argparse, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='Folder with ISOT True.csv and Fake.csv')
    ap.add_argument('--out_csv', default='data/train.csv', help='Output merged CSV')
    args = ap.parse_args()

    true_path = os.path.join(args.data_dir, 'True.csv')
    fake_path = os.path.join(args.data_dir, 'Fake.csv')

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    # --- Assign labels and unify text/title columns ---
    for df, lbl in [(df_true, 'real'), (df_fake, 'fake')]:
        if 'text' not in df.columns:
            if 'content' in df.columns:
                df['text'] = df['content']
            else:
                raise ValueError('Could not find text/content column')
        if 'title' not in df.columns:
            df['title'] = ''
        df['label'] = lbl

    # --- Merge and shuffle ---
    df = pd.concat(
        [df_true[['title', 'text', 'label']], df_fake[['title', 'text', 'label']]],
        ignore_index=True
    )
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    plt.rcParams['figure.figsize'] = (8, 5)
    sns.set(style="whitegrid")

    print("\n===== EDA: Missing Values =====")
    print(df.isnull().sum())

    # Missing-values heatmap
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig("artifacts/missing_heatmap.png")
    plt.close()

    # Drop rows with missing text/label
    before = len(df)
    df = df.dropna(subset=['text', 'label'])
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows with missing text/label.")

    # Label distribution
    sns.countplot(x='label', data=df)
    plt.title("Label Distribution")
    plt.tight_layout()
    plt.savefig("artifacts/label_distribution.png")
    plt.close()

    # Text-length analysis
    df['text_len'] = df['text'].astype(str).str.len()
    sns.boxplot(x=df['text_len'])
    plt.title("Text Length Distribution")
    plt.tight_layout()
    plt.savefig("artifacts/text_length_box.png")
    plt.close()

    sns.histplot(df['text_len'], bins=50)
    plt.title("Text Length Histogram")
    plt.xlabel("Characters per article")
    plt.tight_layout()
    plt.savefig("artifacts/text_length_hist.png")
    plt.close()

    print("[INFO] EDA plots saved under 'artifacts/'.")

    df.to_csv(args.out_csv, index=False)
    print(f"[DONE] Wrote {args.out_csv} with shape {df.shape}")

if __name__ == '__main__':
    main()
