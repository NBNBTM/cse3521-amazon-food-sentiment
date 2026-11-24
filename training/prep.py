import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

SEED = 49
LABELS = ["negative", "neutral", "positive"]

def to_label(s):
    if s <= 2: return "negative"
    if s == 3: return "neutral"
    return "positive"

def load_splits(csv_path):
    raw = pd.read_csv(csv_path)
    sample = raw.sample(n=50000, random_state=SEED)
    df = sample.dropna(subset=["Text"]).copy()
    df["Summary"] = df["Summary"].fillna("")
    df = df.drop_duplicates(subset=["Text"], keep="first").reset_index(drop=True)
    df["doc"] = df["Summary"] + " " + df["Text"]
    df["label"] = df["Score"].map(to_label)
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    fit, val = train_test_split(train, test_size=0.2, stratify=train["label"], random_state=SEED)
    return df, train, test, fit, val

if __name__ == "__main__":
    import sys
    df, train, test, fit, val = load_splits(sys.argv[1])
    print("clean", len(df), "train", len(train), "test", len(test), "fit", len(fit), "val", len(val))
    print(df["label"].value_counts(normalize=True).round(4).to_dict())
